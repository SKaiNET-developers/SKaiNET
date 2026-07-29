#include "skainet_kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native row-major FP32 × FP16 matmul matching the
 * sk.ainet.backend.api.kernel.Fp16MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 *   A, C : FP32 (FloatArray on the JVM side; float* here).
 *          Strides in floats. `a_stride == k` for a contiguous parent.
 *   B    : packed IEEE binary16 (ByteArray on the JVM side; uint8_t* here).
 *          Strides in *bytes*. `b_byte_stride == n * 2` for a contiguous
 *          parent. Each FP16 value is 2 bytes little-endian.
 *
 * The BF16 sibling gets its dequant for free — BF16 is the high half of an
 * FP32, so the conversion is one shift. Binary16 has a narrower exponent and
 * a wider mantissa, so it needs rebiasing, and subnormals need renormalizing.
 * Doing that with branches would cost more than the multiply it feeds, so
 * fp16_to_float below is branch-free: the two special cases are folded in
 * with arithmetic masks, which keeps the inner loop a straight-line sequence
 * the vectorizer can widen (unpack to 32-bit lanes, integer ops, one FMA).
 *
 * Deliberately *not* using _Float16 or F16C intrinsics. The x86_64 build
 * carries no -march flag, so F16C cannot be assumed, and _Float16 without it
 * lowers to libgcc helper calls that are slower than this and block
 * vectorization outright. AArch64 does build with +fp16, but a second code
 * path would double the surface to test for a conversion that is already a
 * handful of integer ops. Runtime ISA dispatch is the place for that, if it
 * ever pays for itself.
 *
 * Iteration order is NOT i-p-j like skainet_bf16_matmul. That order re-decodes
 * every B element once per row of A, which BF16 can afford at one shift per
 * element and this kernel cannot. Instead j is tiled, and within a tile each B
 * row is decoded once into a small stack buffer and then multiplied into all m
 * rows of C. Total decodes drop from m*k*n to k*n while B traffic is unchanged
 * — each element is still read once per tile pass, and the tiles partition n.
 * The tile is sized so that the decoded row plus the m*C rows it touches stay
 * in L1/L2.
 *
 * Accumulation order into any given C element is still p ascending, so this is
 * bit-identical to the i-p-j formulation, not merely close.
 *
 * The same amortization would very likely help skainet_bf16_matmul at m > 1.
 * It is deliberately not applied there in this change: BF16 is the format
 * currently recommended for speed, and changing its measured behaviour belongs
 * in its own change with its own numbers.
 *
 * Caller contract (mirrors skainet_bf16_matmul):
 *  - C is FULLY OVERWRITTEN in the m×n block.
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; defensively treated as no-op.
 *
 * NaN note: a signaling binary16 NaN stays signaling here, matching the JVM
 * Panama kernel. It is not observable — the value is immediately multiplied,
 * and that quiets it.
 */

/* binary16's exponent field once shifted into FP32 position (0x7C00 << 13). */
#define SKAINET_FP16_EXP_FIELD 0x0F800000u
/* (127 - 15) << 23 — the FP32/binary16 exponent bias difference. */
#define SKAINET_FP16_REBIAS 0x38000000u
/* 1 << 23 — one exponent step, lifting a subnormal into the magic binade. */
#define SKAINET_FP16_SUBNORMAL_BUMP 0x00800000u

/*
 * Columns decoded per pass. 512 floats is a 2 KiB stack buffer — small enough
 * to leave the C rows it feeds resident alongside it, large enough that the
 * per-tile loop overhead disappears against the k*m inner work.
 */
#define SKAINET_FP16_TILE 512

static inline float skainet_fp16_to_float(uint16_t h) {
    const uint32_t bits = (uint32_t) h;
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t shifted = (bits & 0x7FFFu) << 13;
    const uint32_t exp_field = shifted & SKAINET_FP16_EXP_FIELD;

    /* 0 or 0xFFFFFFFF, so the corrections below are masks rather than jumps. */
    const uint32_t is_inf_nan =
        (uint32_t) -(int32_t) (exp_field == SKAINET_FP16_EXP_FIELD);
    const uint32_t is_subnormal = (uint32_t) -(int32_t) (exp_field == 0u);

    /* Rebias; Inf/NaN takes a second rebias, which saturates the exponent. */
    const uint32_t biased =
        shifted + SKAINET_FP16_REBIAS + (SKAINET_FP16_REBIAS & is_inf_nan);

    /* Zero and subnormals: bump one exponent step and subtract 2^-14, which
     * makes the FPU renormalize. For a subnormal m * 2^-24 the bumped value is
     * 2^-14 * (1 + m * 2^-10), so the subtraction leaves exactly m * 2^-24;
     * for zero it leaves +0. The sign is reapplied afterwards either way. */
    const uint32_t bumped = biased + SKAINET_FP16_SUBNORMAL_BUMP;
    float renormalized;
    memcpy(&renormalized, &bumped, sizeof(float));
    renormalized -= 0x1p-14f;
    uint32_t renormalized_bits;
    memcpy(&renormalized_bits, &renormalized, sizeof(uint32_t));

    const uint32_t magnitude =
        (renormalized_bits & is_subnormal) | (biased & ~is_subnormal);
    const uint32_t out_bits = sign | magnitude;

    float out;
    memcpy(&out, &out_bits, sizeof(float));
    return out;
}

SKAINET_API void skainet_fp16_matmul(
    const float* SKAINET_RESTRICT a, int32_t a_offset, int32_t a_stride,
    const uint8_t* SKAINET_RESTRICT b, int32_t b_byte_offset, int32_t b_byte_stride,
    float* SKAINET_RESTRICT c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;

    /* Zero the output block. Required by the SPI for k == 0 AND a
     * prerequisite for the i-p-j accumulator below. */
    for (int32_t i = 0; i < m; ++i) {
        float* SKAINET_RESTRICT c_row = c + c_offset + (size_t) i * c_stride;
        for (int32_t j = 0; j < n; ++j) {
            c_row[j] = 0.0f;
        }
    }
    if (k <= 0) return;

    float decoded[SKAINET_FP16_TILE];

    for (int32_t j0 = 0; j0 < n; j0 += SKAINET_FP16_TILE) {
        const int32_t tile =
            (n - j0) < SKAINET_FP16_TILE ? (n - j0) : SKAINET_FP16_TILE;

        for (int32_t p = 0; p < k; ++p) {
            const uint8_t* SKAINET_RESTRICT b_row =
                b + b_byte_offset + (size_t) p * b_byte_stride + (size_t) j0 * 2;

            /* Decode this row's tile once, for all m rows of C below. */
            for (int32_t j = 0; j < tile; ++j) {
                /* Read 2 bytes LE. memcpy is strict-aliasing safe and the
                 * compiler folds it to a single 16-bit load. */
                uint16_t bits;
                memcpy(&bits, b_row + (size_t) j * 2, sizeof(uint16_t));
                decoded[j] = skainet_fp16_to_float(bits);
            }

            for (int32_t i = 0; i < m; ++i) {
                const float a_ip =
                    a[a_offset + (size_t) i * a_stride + p];
                float* SKAINET_RESTRICT c_row =
                    c + c_offset + (size_t) i * c_stride + j0;
                for (int32_t j = 0; j < tile; ++j) {
                    c_row[j] += a_ip * decoded[j];
                }
            }
        }
    }
}
