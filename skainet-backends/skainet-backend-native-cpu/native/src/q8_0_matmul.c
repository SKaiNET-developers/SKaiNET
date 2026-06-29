#include "skainet_kernels.h"
#include "skainet_simd.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native FP32 × Q8_0 matrix-vector matmul matching the
 * sk.ainet.backend.api.kernel.Q8_0MatmulKernel SPI.
 *
 * Block layout (canonical ggml Q8_0):
 *   - bytes 0..1 : FP16 little-endian scale `d`
 *   - bytes 2..33: 32 signed int8 codes
 *
 * Per-block packed weight layout:
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 34
 *
 * Dequant per element: `code * d`. Inner 32-element block dot product
 * auto-vectorizes to vfmadd231ps (x86) / fmla (ARM) under
 * -O3 -ffast-math; scale `d` is folded as a single scalar multiply
 * after the block accumulator, rather than broadcast across every
 * inner-FMA — cheaper, and lets the compiler keep the inner loop tight.
 */

/* Portable FP16 → FP32 conversion. Matches the Kotlin
 * `Q4_KBlockTensorData.halfToFloat` algorithm bit-for-bit so reference
 * tests stay consistent across backends. */
static inline float skainet_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Normalize subnormal. */
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

SKAINET_API void skainet_q8_0_matmul(
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
    const int32_t BYTES_PER_BLOCK = 34;
    const int32_t blocks_per_input_dim = input_dim / BLOCK_SIZE;
    float* SKAINET_RESTRICT out_base = output + output_offset;

    /*
     * Loop order: block OUTER, output row INNER — see q4k_matmul.c for the
     * rationale. The weight is block-major (blockIdx*output_dim + o)*34, so for
     * a fixed block consecutive `o` are 34 bytes apart: weight bytes are read
     * sequentially instead of striding output_dim*34 per step, which on the
     * in-order A55 makes every read a cold cache miss. out_base[o] accumulates
     * across blocks; accumulation order is unchanged ⇒ numerically identical.
     */
    for (int32_t o = 0; o < output_dim; ++o) out_base[o] = 0.0f;

    for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
        const float* SKAINET_RESTRICT input_block =
            input + input_offset + (size_t) block_idx * BLOCK_SIZE;
        const uint8_t* SKAINET_RESTRICT block =
            weight + weight_byte_offset +
            (size_t)(block_idx * output_dim) * BYTES_PER_BLOCK;

        for (int32_t o = 0; o < output_dim; ++o, block += BYTES_PER_BLOCK) {
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            float d = skainet_fp16_to_fp32(d_bits);
            const int8_t* SKAINET_RESTRICT codes = (const int8_t*) (block + 2);
            float block_sum = 0.0f;
#ifdef SKAINET_HAVE_NEON
            /* Activations are FP32, so widen int8 codes to float and FMA
             * (int8 dotprod would need int8 activations — see plan note). */
            float32x4_t accv = vdupq_n_f32(0.0f);
            for (int32_t k = 0; k < BLOCK_SIZE; k += 16) {
                const int8x16_t c8 = vld1q_s8(codes + k);
                const int16x8_t lo16 = vmovl_s8(vget_low_s8(c8));
                const int16x8_t hi16 = vmovl_s8(vget_high_s8(c8));
                const float32x4_t cf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                const float32x4_t cf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                const float32x4_t cf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                const float32x4_t cf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                accv = vfmaq_f32(accv, vld1q_f32(input_block + k),      cf0);
                accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 4),  cf1);
                accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 8),  cf2);
                accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 12), cf3);
            }
            block_sum = skainet_neon_hadd_f32(accv);
#else
            for (int32_t k = 0; k < BLOCK_SIZE; ++k) {
                block_sum += input_block[k] * (float) codes[k];
            }
#endif
            out_base[o] += block_sum * d;
        }
    }
}
