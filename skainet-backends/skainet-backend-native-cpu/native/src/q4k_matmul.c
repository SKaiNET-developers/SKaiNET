#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_cpu_features.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

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
 * Quantize one 256-float input block to symmetric int8 (Q8) with a single
 * per-block scale d_in = maxabs/127, q8[i] = round(in[i]/d_in). Returns d_in
 * (0 if the block is all-zero, with q8 zeroed). Mirrors ggml's block_q8_K
 * activation quantization — the source of the (small, well-understood) error
 * vs the exact float kernel, and what unlocks the int8 dot-product fast path.
 */
static inline float skainet_q8_quantize_block(const float* SKAINET_RESTRICT in, int8_t* SKAINET_RESTRICT q8) {
    float maxabs = 0.0f;
    for (int i = 0; i < Q4K_BLOCK_SIZE; ++i) {
        const float a = fabsf(in[i]);
        if (a > maxabs) maxabs = a;
    }
    if (maxabs == 0.0f) {
        for (int i = 0; i < Q4K_BLOCK_SIZE; ++i) q8[i] = 0;
        return 0.0f;
    }
    const float d_in = maxabs / 127.0f;
    const float inv = 127.0f / maxabs;
    for (int i = 0; i < Q4K_BLOCK_SIZE; ++i) {
        int v = (int) lrintf(in[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        q8[i] = (int8_t) v;
    }
    return d_in;
}

/*
 * One Q4_K block × one output row: the four 64-byte groups, each covering a
 * lo/hi sub-block pair. Extracted so the dotprod body can be compiled twice
 * for the Apple runtime dispatch (see SKAINET_DOTPROD_DISPATCH in
 * skainet_simd.h) — one call per block × output row, so the call overhead is
 * amortized over 128 weight bytes + 64 int8 dots.
 */
#if defined(SKAINET_HAVE_DOTPROD) || defined(SKAINET_DOTPROD_DISPATCH)
SKAINET_DOTPROD_TARGET
static void skainet_q4k_block_dot_dp(
    const uint8_t* SKAINET_RESTRICT qs,
    const int8_t* SKAINET_RESTRICT q8_block,
    const int* SKAINET_RESTRICT scale_idx,
    const int* SKAINET_RESTRICT min_idx,
    int64_t* SKAINET_RESTRICT block_scale_dot,
    int64_t* SKAINET_RESTRICT block_min_sum
) {
    for (int group_j = 0; group_j < 4; ++group_j) {
        const uint8_t* qs_group = qs + group_j * Q4K_SUB_BLOCK_SIZE;
        const int sb_lo = 2 * group_j;
        const int sb_hi = sb_lo + 1;
        const int8_t* q8_lo = q8_block + sb_lo * Q4K_SUB_BLOCK_SIZE;
        const int8_t* q8_hi = q8_block + sb_hi * Q4K_SUB_BLOCK_SIZE;

        int32x4_t acc_dot_lo = vdupq_n_s32(0), acc_dot_hi = vdupq_n_s32(0);
        int32_t acc_sum_lo = 0, acc_sum_hi = 0;
        for (int off = 0; off < Q4K_SUB_BLOCK_SIZE; off += 16) {
            const uint8x16_t packed = vld1q_u8(qs_group + off);
            const int8x16_t code_lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
            const int8x16_t code_hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
            const int8x16_t a_lo = vld1q_s8(q8_lo + off);
            const int8x16_t a_hi = vld1q_s8(q8_hi + off);
            acc_dot_lo = vdotq_s32(acc_dot_lo, code_lo, a_lo);
            acc_dot_hi = vdotq_s32(acc_dot_hi, code_hi, a_hi);
            acc_sum_lo += vaddlvq_s8(a_lo);
            acc_sum_hi += vaddlvq_s8(a_hi);
        }
        const int32_t dot_lo = vaddvq_s32(acc_dot_lo);
        const int32_t dot_hi = vaddvq_s32(acc_dot_hi);

        *block_scale_dot += (int64_t) scale_idx[sb_lo] * dot_lo
                          + (int64_t) scale_idx[sb_hi] * dot_hi;
        *block_min_sum   += (int64_t) min_idx[sb_lo] * acc_sum_lo
                          + (int64_t) min_idx[sb_hi] * acc_sum_hi;
    }
}
#endif

#if !defined(SKAINET_HAVE_DOTPROD) || defined(SKAINET_DOTPROD_DISPATCH)
static void skainet_q4k_block_dot_generic(
    const uint8_t* SKAINET_RESTRICT qs,
    const int8_t* SKAINET_RESTRICT q8_block,
    const int* SKAINET_RESTRICT scale_idx,
    const int* SKAINET_RESTRICT min_idx,
    int64_t* SKAINET_RESTRICT block_scale_dot,
    int64_t* SKAINET_RESTRICT block_min_sum
) {
    for (int group_j = 0; group_j < 4; ++group_j) {
        const uint8_t* qs_group = qs + group_j * Q4K_SUB_BLOCK_SIZE;
        const int sb_lo = 2 * group_j;
        const int sb_hi = sb_lo + 1;
        const int8_t* q8_lo = q8_block + sb_lo * Q4K_SUB_BLOCK_SIZE;
        const int8_t* q8_hi = q8_block + sb_hi * Q4K_SUB_BLOCK_SIZE;

        int32_t dot_lo = 0, sum_lo = 0, dot_hi = 0, sum_hi = 0;
        for (int i = 0; i < Q4K_SUB_BLOCK_SIZE; ++i) {
            const uint8_t pb = qs_group[i];
            const int code_lo = (int)(pb & 0x0F);
            const int code_hi = (int)(pb >> 4);
            const int a_lo = (int) q8_lo[i];
            const int a_hi = (int) q8_hi[i];
            dot_lo += a_lo * code_lo;
            sum_lo += a_lo;
            dot_hi += a_hi * code_hi;
            sum_hi += a_hi;
        }

        *block_scale_dot += (int64_t) scale_idx[sb_lo] * dot_lo
                          + (int64_t) scale_idx[sb_hi] * dot_hi;
        *block_min_sum   += (int64_t) min_idx[sb_lo] * sum_lo
                          + (int64_t) min_idx[sb_hi] * sum_hi;
    }
}
#endif

/*
 * Native Q4_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q4KMatmulKernel SPI contract. Single input row
 * times an `outputDim x inputDim` Q4_K-packed weight laid out
 * (blockIdx * outputDim + o) * 144 bytes.
 *
 * Fused int8 dot path (ggml-style): the input row is quantized to Q8 ONCE per
 * 256-block (reused across all output rows), then each weight sub-block is an
 * int8 dot-product against the Q8 activation:
 *   acc += d_in[b] * ( d * Σ_s scaleIdx[s]*intDot[s] - dMin * Σ_s minIdx[s]*intSum[s] )
 * where intDot[s] = Σ q8[i]*code[i] and intSum[s] = Σ q8[i] over the sub-block.
 * On AArch64 with dotprod (asimddp) the inner dot uses vdotq_s32 (16 int8 MACs
 * per instruction); otherwise a scalar integer fallback (auto-vectorized).
 * The index mapping (groups, lo/hi sub-blocks, input alignment) is identical to
 * the previous float kernel, which was parity-checked against Panama.
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

#ifdef SKAINET_DOTPROD_DISPATCH
    /* One probe per matmul call; cached in skainet_cpu_has_dotprod. */
    const int use_dp = skainet_cpu_has_dotprod();
#endif

    const int32_t blocks_per_input_dim = input_dim / Q4K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    /* Pre-quantize the whole input row to Q8 once (reused across all o). */
    int8_t* q8 = (int8_t*) malloc((size_t) input_dim * sizeof(int8_t));
    float* d_in = (float*) malloc((size_t) blocks_per_input_dim * sizeof(float));
    if (q8 == NULL || d_in == NULL) { free(q8); free(d_in); return; }
    for (int32_t b = 0; b < blocks_per_input_dim; ++b) {
        d_in[b] = skainet_q8_quantize_block(in_base + (size_t) b * Q4K_BLOCK_SIZE,
                                            q8 + (size_t) b * Q4K_BLOCK_SIZE);
    }

    int scale_idx[Q4K_SUB_BLOCKS];
    int min_idx[Q4K_SUB_BLOCKS];

    /*
     * Loop order: block OUTER, output row INNER. The weight is packed
     * block-major — (blockIdx * output_dim + o) * 144 — so for a fixed block,
     * consecutive `o` are exactly 144 bytes apart: the weight bytes are read
     * strictly sequentially (prefetch- and cache-line-friendly). The reverse
     * order (o outer) strides output_dim*144 bytes per step (~295 KB on the
     * down-proj), which on an in-order A55 with small caches makes every weight
     * read a cold miss and dominates runtime regardless of inner-loop compute.
     * out_base[o] is accumulated across blocks (output_dim*4 bytes stays hot in
     * cache); the accumulation order over blocks is unchanged, so this is
     * numerically identical to the o-outer form.
     */
    for (int32_t o = 0; o < output_dim; ++o) out_base[o] = 0.0f;

    for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
        const int8_t* q8_block = q8 + (size_t) block_idx * Q4K_BLOCK_SIZE;
        const float di = d_in[block_idx];
        const uint8_t* block = weight + weight_byte_offset
            + (size_t)(block_idx * output_dim) * Q4K_BYTES_PER_BLOCK;

        for (int32_t o = 0; o < output_dim; ++o, block += Q4K_BYTES_PER_BLOCK) {
            const uint16_t d_bits     = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            const uint16_t d_min_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
            const float d     = skainet_half_to_float(d_bits);
            const float d_min = skainet_half_to_float(d_min_bits);

            skainet_q4k_decode_scales(block + 4, scale_idx, min_idx);

            const uint8_t* qs = block + 16;

            int64_t block_scale_dot = 0;
            int64_t block_min_sum = 0;

#if defined(SKAINET_HAVE_DOTPROD)
            skainet_q4k_block_dot_dp(qs, q8_block, scale_idx, min_idx,
                                     &block_scale_dot, &block_min_sum);
#elif defined(SKAINET_DOTPROD_DISPATCH)
            if (use_dp) {
                skainet_q4k_block_dot_dp(qs, q8_block, scale_idx, min_idx,
                                         &block_scale_dot, &block_min_sum);
            } else {
                skainet_q4k_block_dot_generic(qs, q8_block, scale_idx, min_idx,
                                              &block_scale_dot, &block_min_sum);
            }
#else
            skainet_q4k_block_dot_generic(qs, q8_block, scale_idx, min_idx,
                                          &block_scale_dot, &block_min_sum);
#endif

            out_base[o] += di * (d * (float) block_scale_dot - d_min * (float) block_min_sum);
        }
    }

    free(q8);
    free(d_in);
}

/*
 * Row-major variant (#1189): the weight stays in canonical GGUF file order —
 * (o * blocks_per_row + b) * 144 — so an mmap'd tensor is fed as-is, no
 * relayout copy. Loop order is o OUTER here: each output row's blocks are
 * contiguous on disk, so the weight bytes are still read strictly
 * sequentially; the Q8-quantized activation (input_dim bytes) stays hot
 * across rows. Per-row accumulation order over blocks matches the feed-order
 * kernel's, so results are bit-identical.
 */
SKAINET_API void skainet_q4k_matmul_rm(
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

#ifdef SKAINET_DOTPROD_DISPATCH
    const int use_dp = skainet_cpu_has_dotprod();
#endif

    const int32_t blocks_per_input_dim = input_dim / Q4K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    /* Pre-quantize the whole input row to Q8 once (reused across all o). */
    int8_t* q8 = (int8_t*) malloc((size_t) input_dim * sizeof(int8_t));
    float* d_in = (float*) malloc((size_t) blocks_per_input_dim * sizeof(float));
    if (q8 == NULL || d_in == NULL) { free(q8); free(d_in); return; }
    for (int32_t b = 0; b < blocks_per_input_dim; ++b) {
        d_in[b] = skainet_q8_quantize_block(in_base + (size_t) b * Q4K_BLOCK_SIZE,
                                            q8 + (size_t) b * Q4K_BLOCK_SIZE);
    }

    int scale_idx[Q4K_SUB_BLOCKS];
    int min_idx[Q4K_SUB_BLOCKS];

    const uint8_t* block = weight + weight_byte_offset;
    for (int32_t o = 0; o < output_dim; ++o) {
        float acc = 0.0f;
        for (int32_t block_idx = 0; block_idx < blocks_per_input_dim;
             ++block_idx, block += Q4K_BYTES_PER_BLOCK) {
            const int8_t* q8_block = q8 + (size_t) block_idx * Q4K_BLOCK_SIZE;
            const float di = d_in[block_idx];

            const uint16_t d_bits     = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            const uint16_t d_min_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
            const float d     = skainet_half_to_float(d_bits);
            const float d_min = skainet_half_to_float(d_min_bits);

            skainet_q4k_decode_scales(block + 4, scale_idx, min_idx);

            const uint8_t* qs = block + 16;

            int64_t block_scale_dot = 0;
            int64_t block_min_sum = 0;

#if defined(SKAINET_HAVE_DOTPROD)
            skainet_q4k_block_dot_dp(qs, q8_block, scale_idx, min_idx,
                                     &block_scale_dot, &block_min_sum);
#elif defined(SKAINET_DOTPROD_DISPATCH)
            if (use_dp) {
                skainet_q4k_block_dot_dp(qs, q8_block, scale_idx, min_idx,
                                         &block_scale_dot, &block_min_sum);
            } else {
                skainet_q4k_block_dot_generic(qs, q8_block, scale_idx, min_idx,
                                              &block_scale_dot, &block_min_sum);
            }
#else
            skainet_q4k_block_dot_generic(qs, q8_block, scale_idx, min_idx,
                                          &block_scale_dot, &block_min_sum);
#endif

            acc += di * (d * (float) block_scale_dot - d_min * (float) block_min_sum);
        }
        out_base[o] = acc;
    }

    free(q8);
    free(d_in);
}
