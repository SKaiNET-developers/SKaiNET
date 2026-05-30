#include "skainet_kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native FP32 × Q4_0 matrix-vector matmul matching the
 * sk.ainet.backend.api.kernel.Q4_0MatmulKernel SPI.
 *
 * Block layout (canonical ggml Q4_0, 32 elements, 18 bytes):
 *   - bytes 0..1  : FP16 little-endian scale `d`
 *   - bytes 2..17 : 16 bytes packing 32 4-bit codes in the *split*
 *     layout — low nibbles decode elements 0..15, high nibbles decode
 *     elements 16..31.
 *
 * Per-block packed weight layout:
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 18
 *
 * Dequant per element: `(code - 8) * d`. The `- 8` bias centres the
 * unsigned 4-bit code. Scale `d` is folded once after the block
 * accumulator (cheaper than broadcasting it across every inner FMA).
 */

/* Portable FP16 → FP32 conversion. Matches the Kotlin
 * `Q4_0BlockTensorData.halfToFloat` algorithm bit-for-bit. */
static inline float skainet_q4_0_fp16_to_fp32(uint16_t h) {
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

SKAINET_API void skainet_q4_0_matmul(
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
    const int32_t BYTES_PER_BLOCK = 18;
    const int32_t blocks_per_input_dim = input_dim / BLOCK_SIZE;

    for (int32_t o = 0; o < output_dim; ++o) {
        float acc = 0.0f;
        for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
            const uint8_t* SKAINET_RESTRICT block =
                weight + weight_byte_offset +
                (size_t)(block_idx * output_dim + o) * BYTES_PER_BLOCK;
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            float d = skainet_q4_0_fp16_to_fp32(d_bits);
            const uint8_t* SKAINET_RESTRICT codes = block + 2;
            const float* SKAINET_RESTRICT input_block =
                input + input_offset + (size_t) block_idx * BLOCK_SIZE;
            float block_sum = 0.0f;
            for (int32_t k = 0; k < 16; ++k) {
                int32_t lo = (int32_t)(codes[k] & 0x0F) - 8;
                int32_t hi = (int32_t)(codes[k] >> 4) - 8;
                block_sum += input_block[k] * (float) lo;
                block_sum += input_block[k + 16] * (float) hi;
            }
            acc += block_sum * d;
        }
        output[output_offset + o] = acc;
    }
}
