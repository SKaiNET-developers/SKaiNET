#include "skainet_kernels.h"

#include <stddef.h>
#include <stdint.h>

#define Q4K_BLOCK_SIZE       256
#define Q4K_SUB_BLOCK_SIZE    32
#define Q4K_SUB_BLOCKS         8
#define Q4K_BYTES_PER_BLOCK  144

/*
 * IEEE 754 binary16 (LE byte order) -> binary32 conversion.
 * Mirrors PanamaVectorQ4KMatmulKernel.halfToFloat byte-for-byte.
 */
static inline float skainet_half_to_float(uint16_t hbits) {
    const uint32_t sign = (hbits >> 15) & 0x1u;
    const uint32_t exp  = (hbits >> 10) & 0x1Fu;
    const uint32_t frac =  hbits        & 0x3FFu;

    if (exp == 0u) {
        if (frac == 0u) {
            union { uint32_t u; float f; } v = { sign << 31 };
            return v.f;
        }
        float f = ((float) frac) / 1024.0f * (1.0f / 16384.0f);
        return sign ? -f : f;
    }
    if (exp == 0x1Fu) {
        union { uint32_t u; float f; } v;
        v.u = (sign << 31) | 0x7F800000u | (frac ? 0x00400000u : 0u);
        return v.f;
    }
    union { uint32_t u; float f; } v;
    v.u = (sign << 31) | ((exp - 15u + 127u) << 23) | (frac << 13);
    return v.f;
}

/*
 * ggml's get_scale_min_k4 unmix for the 12-byte packed sub-scale region
 * (bytes 4..15 of a Q4_K block). Same logic as the Kotlin reference.
 */
static inline void skainet_q4k_decode_scales(
    const uint8_t* scales,
    int* scale_idx,
    int* min_idx
) {
    for (int sb = 0; sb < 4; ++sb) {
        scale_idx[sb] = scales[sb]     & 0x3F;
        min_idx[sb]   = scales[sb + 4] & 0x3F;
    }
    for (int sb = 4; sb < 8; ++sb) {
        const int low4_s  = scales[sb + 4] & 0x0F;
        const int high2_s = (scales[sb - 4] >> 6) & 0x03;
        scale_idx[sb] = low4_s | (high2_s << 4);

        const int low4_m  = (scales[sb + 4] >> 4) & 0x0F;
        const int high2_m = (scales[sb] >> 6) & 0x03;
        min_idx[sb] = low4_m | (high2_m << 4);
    }
}

/*
 * Native Q4_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q4KMatmulKernel SPI contract. Single
 * input row times an `outputDim x inputDim` Q4_K-packed weight tensor
 * laid out (blockIdx * outputDim + o) * 144 bytes.
 *
 * Lazy-dmin pattern: per sub-block accumulate
 *   codeSum[s] = sum_i input[i] * code[i]
 *   inputSum[s] = sum_i input[i]
 * and combine once via
 *   acc += d * scaleIdx[s] * codeSum[s] - dMin * minIdx[s] * inputSum[s]
 *
 * Scalar single-threaded for PR 2; the tight inner loop is
 * straight-line FP arithmetic so -O3 auto-vectorizes the
 * codeSum/inputSum accumulators on AVX2/NEON.
 */
SKAINET_API void skainet_q4k_matmul(
    const float* SKAINET_RESTRICT input,
    int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* SKAINET_RESTRICT output,
    int32_t output_offset
) {
    if (output_dim <= 0 || input_dim <= 0) return;

    const int32_t blocks_per_input_dim = input_dim / Q4K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    int scale_idx[Q4K_SUB_BLOCKS];
    int min_idx[Q4K_SUB_BLOCKS];

    for (int32_t o = 0; o < output_dim; ++o) {
        float acc = 0.0f;

        for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
            const uint8_t* block = weight + weight_byte_offset
                + (size_t)(block_idx * output_dim + o) * Q4K_BYTES_PER_BLOCK;

            /* d, dMin (FP16 LE -> FP32). */
            const uint16_t d_bits     = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            const uint16_t d_min_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
            const float d     = skainet_half_to_float(d_bits);
            const float d_min = skainet_half_to_float(d_min_bits);

            /* 12 bytes of packed (scaleIdx, minIdx) -> 8 ints each. */
            skainet_q4k_decode_scales(block + 4, scale_idx, min_idx);

            const uint8_t* qs = block + 16;
            const float* in_block = in_base + (size_t) block_idx * Q4K_BLOCK_SIZE;

            /* 4 strided qs groups; group j carries sub-blocks 2j (lo) and 2j+1 (hi). */
            for (int group_j = 0; group_j < 4; ++group_j) {
                const uint8_t* qs_group   = qs + group_j * Q4K_SUB_BLOCK_SIZE;
                const int sb_lo = 2 * group_j;
                const int sb_hi = sb_lo + 1;
                const float* in_lo = in_block + sb_lo * Q4K_SUB_BLOCK_SIZE;
                const float* in_hi = in_block + sb_hi * Q4K_SUB_BLOCK_SIZE;

                float code_sum_lo = 0.0f, input_sum_lo = 0.0f;
                float code_sum_hi = 0.0f, input_sum_hi = 0.0f;

                /* 32 iterations — auto-vectorizes cleanly under -O3. */
                for (int i = 0; i < Q4K_SUB_BLOCK_SIZE; ++i) {
                    const uint8_t b = qs_group[i];
                    const float code_lo = (float)(b & 0x0F);
                    const float code_hi = (float)(b >> 4);
                    const float v_lo = in_lo[i];
                    const float v_hi = in_hi[i];
                    code_sum_lo  += v_lo * code_lo;
                    input_sum_lo += v_lo;
                    code_sum_hi  += v_hi * code_hi;
                    input_sum_hi += v_hi;
                }

                const float scale_lo  = d     * (float) scale_idx[sb_lo];
                const float offset_lo = d_min * (float) min_idx[sb_lo];
                const float scale_hi  = d     * (float) scale_idx[sb_hi];
                const float offset_hi = d_min * (float) min_idx[sb_hi];
                acc += code_sum_lo * scale_lo - input_sum_lo * offset_lo;
                acc += code_sum_hi * scale_hi - input_sum_hi * offset_hi;
            }
        }

        out_base[o] = acc;
    }
}
