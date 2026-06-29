#include "skainet_kernels.h"
#include "skainet_simd.h"

#include <stddef.h>
#include <stdint.h>

#define Q5K_BLOCK_SIZE       256
#define Q5K_SUB_BLOCK_SIZE    32
#define Q5K_SUB_BLOCKS         8
#define Q5K_BYTES_PER_BLOCK  176
#define Q5K_QH_OFFSET         16
#define Q5K_QS_OFFSET         48

/*
 * IEEE 754 binary16 (LE byte order) -> binary32 conversion.
 * Mirrors PanamaVectorQ5_KMatmulKernel.halfToFloat / the Q4_K kernel
 * byte-for-byte (kept scalar to preserve bit-exact FP16 parity).
 */
static inline float skainet_q5k_half_to_float(uint16_t hbits) {
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
 * (bytes 4..15). Identical to Q4_K.
 */
static inline void skainet_q5k_decode_scales(
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
 * Native Q5_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q5KMatmulKernel SPI contract. Single
 * input row times an `outputDim x inputDim` Q5_K-packed weight tensor
 * laid out (blockIdx * outputDim + o) * 176 bytes.
 *
 * Q5_K extends Q4_K with a 32-byte `qh` high-bit plane: the 5-bit code
 * is `lowNibble | (fifthBit << 4)`, where the low nibble lives in `qs`
 * (same strided layout as Q4_K) and the 5th bit is bit (2*group) of
 * qh[l] for the low sub-block, (2*group + 1) for the high sub-block.
 *
 * Lazy-dmin pattern: per sub-block accumulate
 *   codeSum[s] = sum_i input[i] * code[i]
 *   inputSum[s] = sum_i input[i]
 * and combine once via
 *   acc += d * scaleIdx[s] * codeSum[s] - dMin * minIdx[s] * inputSum[s]
 *
 * Scalar single-threaded; the tight inner loop is straight-line FP
 * arithmetic so -O3 auto-vectorizes on AVX2/NEON. A hand-written NEON
 * path is layered on behind __ARM_NEON in a later PR.
 */
SKAINET_API void skainet_q5k_matmul(
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

    const int32_t blocks_per_input_dim = input_dim / Q5K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    int scale_idx[Q5K_SUB_BLOCKS];
    int min_idx[Q5K_SUB_BLOCKS];

    /*
     * Loop order: block OUTER, output row INNER — see q4k_matmul.c for the
     * rationale. The weight is block-major (blockIdx*output_dim + o)*176, so for
     * a fixed block consecutive `o` are 176 bytes apart: weight bytes are read
     * sequentially (cache/prefetch friendly) instead of striding output_dim*176
     * per step, which on the in-order A55 makes every read a cold miss.
     * out_base[o] accumulates across blocks (a per-o register `acc` holds the
     * inner sum); accumulation order over blocks is unchanged ⇒ numerically
     * identical to the o-outer form.
     */
    for (int32_t o = 0; o < output_dim; ++o) out_base[o] = 0.0f;

    for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
        const float* in_block = in_base + (size_t) block_idx * Q5K_BLOCK_SIZE;
        const uint8_t* block = weight + weight_byte_offset
            + (size_t)(block_idx * output_dim) * Q5K_BYTES_PER_BLOCK;

        for (int32_t o = 0; o < output_dim; ++o, block += Q5K_BYTES_PER_BLOCK) {
            float acc = 0.0f;

            /* d, dMin (FP16 LE -> FP32). */
            const uint16_t d_bits     = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            const uint16_t d_min_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
            const float d     = skainet_q5k_half_to_float(d_bits);
            const float d_min = skainet_q5k_half_to_float(d_min_bits);

            /* 12 bytes of packed (scaleIdx, minIdx) -> 8 ints each. */
            skainet_q5k_decode_scales(block + 4, scale_idx, min_idx);

            const uint8_t* qh = block + Q5K_QH_OFFSET;
            const uint8_t* qs = block + Q5K_QS_OFFSET;

            /* 4 strided qs groups; group j carries sub-blocks 2j (lo) and 2j+1 (hi). */
            for (int group_j = 0; group_j < 4; ++group_j) {
                const uint8_t* qs_group = qs + group_j * Q5K_SUB_BLOCK_SIZE;
                const int sb_lo = 2 * group_j;
                const int sb_hi = sb_lo + 1;
                const int bit_lo = 2 * group_j;
                const int bit_hi = 2 * group_j + 1;
                const float* in_lo = in_block + sb_lo * Q5K_SUB_BLOCK_SIZE;
                const float* in_hi = in_block + sb_hi * Q5K_SUB_BLOCK_SIZE;

                float code_sum_lo = 0.0f, input_sum_lo = 0.0f;
                float code_sum_hi = 0.0f, input_sum_hi = 0.0f;

#ifdef SKAINET_HAVE_NEON
                /* Variable right-shift via vshlq_u8 with a negative count
                 * (bit_lo/bit_hi are runtime values, so vshrq_n_u8's
                 * immediate form can't be used). */
                const int8x16_t shr_lo = vdupq_n_s8(-(int8_t) bit_lo);
                const int8x16_t shr_hi = vdupq_n_s8(-(int8_t) bit_hi);
                float32x4_t cacc_lo = vdupq_n_f32(0.0f), iacc_lo = vdupq_n_f32(0.0f);
                float32x4_t cacc_hi = vdupq_n_f32(0.0f), iacc_hi = vdupq_n_f32(0.0f);
                for (int off = 0; off < Q5K_SUB_BLOCK_SIZE; off += 16) {
                    const uint8x16_t packed = vld1q_u8(qs_group + off);
                    const uint8x16_t qhv = vld1q_u8(qh + off);
                    const uint8x16_t lo_nib = vandq_u8(packed, vdupq_n_u8(0x0F));
                    const uint8x16_t hi_nib = vshrq_n_u8(packed, 4);
                    /* 5th bit -> bit 4 of the code byte. */
                    const uint8x16_t fifth_lo =
                        vshlq_n_u8(vandq_u8(vshlq_u8(qhv, shr_lo), vdupq_n_u8(0x01)), 4);
                    const uint8x16_t fifth_hi =
                        vshlq_n_u8(vandq_u8(vshlq_u8(qhv, shr_hi), vdupq_n_u8(0x01)), 4);
                    const uint8x16_t code_lo = vorrq_u8(lo_nib, fifth_lo);
                    const uint8x16_t code_hi = vorrq_u8(hi_nib, fifth_hi);
                    float32x4_t cl[4], ch[4];
                    skainet_neon_u8x16_to_f32x4x4(code_lo, cl);
                    skainet_neon_u8x16_to_f32x4x4(code_hi, ch);
                    for (int q = 0; q < 4; ++q) {
                        const float32x4_t v_lo = vld1q_f32(in_lo + off + q * 4);
                        const float32x4_t v_hi = vld1q_f32(in_hi + off + q * 4);
                        cacc_lo = vfmaq_f32(cacc_lo, v_lo, cl[q]);
                        iacc_lo = vaddq_f32(iacc_lo, v_lo);
                        cacc_hi = vfmaq_f32(cacc_hi, v_hi, ch[q]);
                        iacc_hi = vaddq_f32(iacc_hi, v_hi);
                    }
                }
                code_sum_lo = skainet_neon_hadd_f32(cacc_lo);
                input_sum_lo = skainet_neon_hadd_f32(iacc_lo);
                code_sum_hi = skainet_neon_hadd_f32(cacc_hi);
                input_sum_hi = skainet_neon_hadd_f32(iacc_hi);
#else
                /* 32 iterations — auto-vectorizes cleanly under -O3. */
                for (int i = 0; i < Q5K_SUB_BLOCK_SIZE; ++i) {
                    const uint8_t b = qs_group[i];
                    const uint8_t h = qh[i];
                    const float code_lo = (float)((b & 0x0F) | (((h >> bit_lo) & 0x01) << 4));
                    const float code_hi = (float)((b >> 4)   | (((h >> bit_hi) & 0x01) << 4));
                    const float v_lo = in_lo[i];
                    const float v_hi = in_hi[i];
                    code_sum_lo  += v_lo * code_lo;
                    input_sum_lo += v_lo;
                    code_sum_hi  += v_hi * code_hi;
                    input_sum_hi += v_hi;
                }
#endif

                const float scale_lo  = d     * (float) scale_idx[sb_lo];
                const float offset_lo = d_min * (float) min_idx[sb_lo];
                const float scale_hi  = d     * (float) scale_idx[sb_hi];
                const float offset_hi = d_min * (float) min_idx[sb_hi];
                acc += code_sum_lo * scale_lo - input_sum_lo * offset_lo;
                acc += code_sum_hi * scale_hi - input_sum_hi * offset_hi;
            }

            out_base[o] += acc;
        }
    }
}
