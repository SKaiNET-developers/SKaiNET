#include "skainet_kernels.h"
#include "skainet_simd.h"

#include <stddef.h>
#include <stdint.h>

#define Q6K_BLOCK_SIZE       256
#define Q6K_BYTES_PER_BLOCK  210
#define Q6K_QL_OFFSET          0
#define Q6K_QH_OFFSET        128
#define Q6K_SCALES_OFFSET    192
#define Q6K_D_OFFSET         208

/*
 * IEEE 754 binary16 (LE byte order) -> binary32 conversion.
 * Byte-for-byte identical to the Q5_K / Q4_K converter (kept scalar to
 * preserve bit-exact FP16 parity with the Panama / scalar references).
 */
static inline float skainet_q6k_half_to_float(uint16_t hbits) {
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
 * Dequantize one 256-element Q6_K super-block into scratch[256].
 * Direct transcription of ScalarQ6_KMatmulKernel.dequantBlock /
 * ggml dequantize_row_q6_K: two 128-element halves, each split into two
 * 16-element scale groups carrying four strided sub-codes (q1..q4).
 *
 * The 6-bit code is `lowNibble(ql) | (twoHighBits(qh) << 4)`, biased by
 * -32, and `scales` are SIGNED int8. Per-element value = d * scale * code.
 */
static inline void skainet_q6k_dequant_block(const uint8_t* SKAINET_RESTRICT block,
                                             float* SKAINET_RESTRICT scratch) {
    const uint8_t* ql0 = block + Q6K_QL_OFFSET;
    const uint8_t* qh0 = block + Q6K_QH_OFFSET;
    const int8_t*  sc0 = (const int8_t*)(block + Q6K_SCALES_OFFSET);
    const uint16_t d_bits = (uint16_t) block[Q6K_D_OFFSET]
        | ((uint16_t) block[Q6K_D_OFFSET + 1] << 8);
    const float d = skainet_q6k_half_to_float(d_bits);

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql = ql0 + half * 64;
        const uint8_t* qh = qh0 + half * 32;
        const int8_t*  sc = sc0 + half * 8;
        float* out = scratch + half * 128;
        for (int is = 0; is < 2; ++is) {
            const float sc1 = d * (float) sc[is + 0];
            const float sc2 = d * (float) sc[is + 2];
            const float sc3 = d * (float) sc[is + 4];
            const float sc4 = d * (float) sc[is + 6];
            const int l_start = is * 16;
            for (int l = l_start; l < l_start + 16; ++l) {
                const int q_l0  = ql[l];
                const int q_l32 = ql[l + 32];
                const int q_h   = qh[l];
                const int q1 = ((q_l0  & 0x0F) | ((q_h        & 0x03) << 4)) - 32;
                const int q2 = ((q_l32 & 0x0F) | (((q_h >> 2) & 0x03) << 4)) - 32;
                const int q3 = ((q_l0  >> 4)   | (((q_h >> 4) & 0x03) << 4)) - 32;
                const int q4 = ((q_l32 >> 4)   | (((q_h >> 6) & 0x03) << 4)) - 32;
                out[l +  0] = sc1 * (float) q1;
                out[l + 32] = sc2 * (float) q2;
                out[l + 64] = sc3 * (float) q3;
                out[l + 96] = sc4 * (float) q4;
            }
        }
    }
}

/*
 * Native Q6_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q6KMatmulKernel SPI contract. A single
 * input row times an `outputDim x inputDim` Q6_K-packed weight tensor
 * laid out (blockIdx * outputDim + o) * 210 bytes.
 *
 * The 6-bit bit-assembly is kept scalar (cheap byte shuffling that the
 * compiler auto-vectorizes under -O3) and materialized into a 256-float
 * scratch block; the hot dot product against the input window is the
 * NEON path (vfmaq_f32 + horizontal add) behind __ARM_NEON. On non-ARM
 * targets the dot is a straight-line loop that auto-vectorizes too.
 */
SKAINET_API void skainet_q6k_matmul(
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

    const int32_t blocks_per_input_dim = input_dim / Q6K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    float scratch[Q6K_BLOCK_SIZE];

    for (int32_t o = 0; o < output_dim; ++o) {
        float acc = 0.0f;

        for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
            const uint8_t* block = weight + weight_byte_offset
                + (size_t)(block_idx * output_dim + o) * Q6K_BYTES_PER_BLOCK;

            skainet_q6k_dequant_block(block, scratch);

            const float* in_block = in_base + (size_t) block_idx * Q6K_BLOCK_SIZE;

#ifdef SKAINET_HAVE_NEON
            float32x4_t vacc = vdupq_n_f32(0.0f);
            for (int i = 0; i < Q6K_BLOCK_SIZE; i += 4) {
                const float32x4_t vi = vld1q_f32(in_block + i);
                const float32x4_t vw = vld1q_f32(scratch + i);
                vacc = vfmaq_f32(vacc, vi, vw);
            }
            acc += skainet_neon_hadd_f32(vacc);
#else
            for (int i = 0; i < Q6K_BLOCK_SIZE; ++i) {
                acc += in_block[i] * scratch[i];
            }
#endif
        }

        out_base[o] = acc;
    }
}
