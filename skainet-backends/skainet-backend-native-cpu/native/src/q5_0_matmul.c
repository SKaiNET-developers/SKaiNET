#include "skainet_kernels.h"
#include "skainet_simd.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native FP32 × Q5_0 matrix-vector matmul matching the
 * sk.ainet.backend.api.kernel.Q5_0MatmulKernel SPI.
 *
 * Block layout (canonical ggml Q5_0, 32 elements, 22 bytes):
 *   - bytes 0..1  : FP16 little-endian scale `d`
 *   - bytes 2..5  : `qh` — 32-bit little-endian high-bit plane, bit j is
 *     the fifth bit of element j
 *   - bytes 6..21 : 16 bytes packing 32 4-bit codes in the *split*
 *     layout — low nibbles decode elements 0..15, high nibbles decode
 *     elements 16..31.
 *
 * Per-block packed weight layout:
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 22
 *
 * Dequant per element: `(code - 16) * d` with
 * `code = nibble | (fifth_bit << 4)` (unsigned, 0..31). The kernel uses
 * the algebraic split
 *
 *   sum_j x_j * d * (code_j - 16) = d * (dot(x, code) - 16 * sum(x))
 *
 * so the inner loop accumulates the *unsigned* code dot product and the
 * per-block input sum is hoisted OUT of the output-row loop (it only
 * depends on the activations, not on `o`). Scale folding happens once
 * per block, exactly like q4_0/q8_0.
 *
 * Loop order: block OUTER, output row INNER — see q8_0_matmul.c for the
 * rationale (sequential weight reads; per-row accumulation order stays
 * ascending-block, so results don't depend on output_dim).
 *
 * NEON path (SKAINET_HAVE_NEON): plain NEON only — no dotprod/i8mm
 * requirement, so the body runs on every AArch64 core. The high-bit
 * plane is expanded with a per-lane vtstq_u8 against a {1,2,4,…,128}
 * bitmask after broadcasting each `qh` byte to its 8 lanes, masked to
 * the fifth-bit value (16) and OR'd onto the nibbles; the resulting
 * unsigned 5-bit codes widen to f32 FMA lanes via
 * skainet_neon_u8x16_to_f32x4x4, the same structure as q8_0/q4_0.
 */

/* Portable FP16 → FP32 conversion. Matches the Kotlin `decodeHalf`
 * algorithm bit-for-bit. */
static inline float skainet_q5_0_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &bits, sizeof(r));
    return r;
}

SKAINET_API void skainet_q5_0_matmul(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    if (output_dim <= 0) return;
    if (input_dim <= 0) {
        for (int32_t o = 0; o < output_dim; ++o) {
            output[output_offset + o] = 0.0f;
        }
        return;
    }

    const int32_t BLOCK_SIZE = 32;
    const int32_t BYTES_PER_BLOCK = 22;
    const int32_t blocks_per_input_dim = input_dim / BLOCK_SIZE;
    float* SKAINET_RESTRICT out_base = output + output_offset;

    for (int32_t o = 0; o < output_dim; ++o) out_base[o] = 0.0f;

    for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
        const float* SKAINET_RESTRICT input_block =
            input + input_offset + (size_t) block_idx * BLOCK_SIZE;
        const uint8_t* SKAINET_RESTRICT block =
            weight + weight_byte_offset +
            (size_t)(block_idx * output_dim) * BYTES_PER_BLOCK;

        /* Depends only on the activations — hoisted out of the o-loop. */
        float input_sum = 0.0f;
        for (int32_t k = 0; k < BLOCK_SIZE; ++k) input_sum += input_block[k];

        for (int32_t o = 0; o < output_dim; ++o, block += BYTES_PER_BLOCK) {
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            float d = skainet_q5_0_fp16_to_fp32(d_bits);
            const uint8_t* SKAINET_RESTRICT qh = block + 2;
            const uint8_t* SKAINET_RESTRICT qs = block + 6;
            float code_dot = 0.0f;
#ifdef SKAINET_HAVE_NEON
            /* Broadcast each qh byte to its 8 lanes, test the per-lane bit,
             * mask to the fifth-bit value 16 and OR onto the nibbles. */
            const uint8x16_t bitmask = vcombine_u8(
                vcreate_u8(0x8040201008040201ULL),
                vcreate_u8(0x8040201008040201ULL));
            const uint8x8_t qh_bytes = vreinterpret_u8_u32(vdup_n_u32(
                (uint32_t) qh[0] | ((uint32_t) qh[1] << 8) |
                ((uint32_t) qh[2] << 16) | ((uint32_t) qh[3] << 24)));
            const uint8x16_t qh_lo = vcombine_u8(
                vdup_lane_u8(qh_bytes, 0), vdup_lane_u8(qh_bytes, 1));
            const uint8x16_t qh_hi = vcombine_u8(
                vdup_lane_u8(qh_bytes, 2), vdup_lane_u8(qh_bytes, 3));
            const uint8x16_t fifth = vdupq_n_u8(0x10);
            const uint8x16_t fifth_lo = vandq_u8(vtstq_u8(qh_lo, bitmask), fifth);
            const uint8x16_t fifth_hi = vandq_u8(vtstq_u8(qh_hi, bitmask), fifth);

            const uint8x16_t packed = vld1q_u8(qs);
            const uint8x16_t code_lo = vorrq_u8(
                vandq_u8(packed, vdupq_n_u8(0x0F)), fifth_lo); /* elems 0..15  */
            const uint8x16_t code_hi = vorrq_u8(
                vshrq_n_u8(packed, 4), fifth_hi);              /* elems 16..31 */

            float32x4_t lo_f[4];
            float32x4_t hi_f[4];
            skainet_neon_u8x16_to_f32x4x4(code_lo, lo_f);
            skainet_neon_u8x16_to_f32x4x4(code_hi, hi_f);
            float32x4_t accv = vdupq_n_f32(0.0f);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 0),  lo_f[0]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 4),  lo_f[1]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 8),  lo_f[2]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 12), lo_f[3]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 16), hi_f[0]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 20), hi_f[1]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 24), hi_f[2]);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 28), hi_f[3]);
            code_dot = skainet_neon_hadd_f32(accv);
#else
            uint32_t qh32 = (uint32_t) qh[0] | ((uint32_t) qh[1] << 8) |
                            ((uint32_t) qh[2] << 16) | ((uint32_t) qh[3] << 24);
            for (int32_t k = 0; k < 16; ++k) {
                int32_t lo = (int32_t)(qs[k] & 0x0F) | (int32_t)(((qh32 >> k) & 1u) << 4);
                int32_t hi = (int32_t)(qs[k] >> 4) | (int32_t)(((qh32 >> (k + 16)) & 1u) << 4);
                code_dot += input_block[k] * (float) lo;
                code_dot += input_block[k + 16] * (float) hi;
            }
#endif
            out_base[o] += d * (code_dot - 16.0f * input_sum);
        }
    }
}
