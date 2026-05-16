#include "skainet_kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native row-major FP32 × BF16 matmul matching the
 * sk.ainet.backend.api.kernel.Bf16MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 *   A, C : FP32 (FloatArray on the JVM side; float* here).
 *          Strides in floats. `a_stride == k` for a contiguous parent.
 *   B    : packed BF16 (ByteArray on the JVM side; uint8_t* here).
 *          Strides in *bytes*. `b_byte_stride == n * 2` for a contiguous
 *          parent. Each BF16 value is 2 bytes little-endian.
 *
 * BF16 → FP32 conversion is the bit-shift identity
 *   float_bits = ((uint32_t) bf16_bits) << 16
 * (BF16 shares the FP32 sign and exponent layout; only the trailing 16
 *  mantissa bits are discarded).
 *
 * Iteration order is i-p-j (outer-product into rows of C). The inner
 * `c[j] += a_ip * bf16_to_float(b[j])` loop streams two contiguous
 * arrays — auto-vectorizes under -O3 -ffast-math into vfmadd231ps
 * (x86_64) / fmla (AArch64). On ARMv8.6-A+ a future pass can swap the
 * scalar dequant for a `bfdot`/`bfmmla` intrinsic kernel; that lives
 * behind a runtime feature check and is out of scope here.
 *
 * Caller contract (mirrors skainet_fp32_matmul):
 *  - C is FULLY OVERWRITTEN in the m×n block.
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; defensively treated as no-op.
 */
SKAINET_API void skainet_bf16_matmul(
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

    for (int32_t i = 0; i < m; ++i) {
        const float* SKAINET_RESTRICT a_row = a + a_offset + (size_t) i * a_stride;
        float* SKAINET_RESTRICT c_row = c + c_offset + (size_t) i * c_stride;
        for (int32_t p = 0; p < k; ++p) {
            const float a_ip = a_row[p];
            const uint8_t* SKAINET_RESTRICT b_row =
                b + b_byte_offset + (size_t) p * b_byte_stride;
            for (int32_t j = 0; j < n; ++j) {
                /* Read 2 bytes LE. memcpy is strict-aliasing safe and the
                 * compiler folds it to a single 16-bit load. */
                uint16_t bits;
                memcpy(&bits, b_row + (size_t) j * 2, sizeof(uint16_t));
                uint32_t fp32_bits = ((uint32_t) bits) << 16;
                float b_pj;
                memcpy(&b_pj, &fp32_bits, sizeof(float));
                c_row[j] += a_ip * b_pj;
            }
        }
    }
}
