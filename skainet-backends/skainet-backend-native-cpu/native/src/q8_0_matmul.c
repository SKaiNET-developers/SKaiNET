#include "skainet_kernels.h"

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

    for (int32_t o = 0; o < output_dim; ++o) {
        float acc = 0.0f;
        for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
            const uint8_t* SKAINET_RESTRICT block =
                weight + weight_byte_offset +
                (size_t)(block_idx * output_dim + o) * BYTES_PER_BLOCK;
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            float d = skainet_fp16_to_fp32(d_bits);
            const int8_t* SKAINET_RESTRICT codes = (const int8_t*) (block + 2);
            const float* SKAINET_RESTRICT input_block =
                input + input_offset + (size_t) block_idx * BLOCK_SIZE;
            float block_sum = 0.0f;
            for (int32_t k = 0; k < BLOCK_SIZE; ++k) {
                block_sum += input_block[k] * (float) codes[k];
            }
            acc += block_sum * d;
        }
        output[output_offset + o] = acc;
    }
}
