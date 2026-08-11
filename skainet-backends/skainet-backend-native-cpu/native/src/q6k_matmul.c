#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_cpu_features.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

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
 * Quantize one 256-float input block to symmetric int8 (Q8), d_in = maxabs/127,
 * q8[i] = round(in[i]/d_in). Mirrors q4k_matmul.c's activation quant (ggml
 * block_q8_K style) — the source of the small (~1-3%) error vs the exact float
 * kernel and what unlocks the int8 dot path. Returns d_in (0 + zeroed q8 if the
 * block is all-zero).
 */
static inline float skainet_q6k_q8_quantize_block(const float* SKAINET_RESTRICT in,
                                                  int8_t* SKAINET_RESTRICT q8) {
    float maxabs = 0.0f;
    for (int i = 0; i < Q6K_BLOCK_SIZE; ++i) {
        const float a = in[i] < 0.0f ? -in[i] : in[i];
        if (a > maxabs) maxabs = a;
    }
    if (maxabs == 0.0f) {
        for (int i = 0; i < Q6K_BLOCK_SIZE; ++i) q8[i] = 0;
        return 0.0f;
    }
    const float d_in = maxabs / 127.0f;
    const float inv = 127.0f / maxabs;
    for (int i = 0; i < Q6K_BLOCK_SIZE; ++i) {
        int v = (int) lrintf(in[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        q8[i] = (int8_t) v;
    }
    return d_in;
}

/*
 * Unpack one 256-element Q6_K super-block into CENTERED int8 codes[256] (the
 * 6-bit code biased by -32, range [-32, 31]) in natural element order — i.e.
 * codes[i] pairs with input[i]. Same bit layout as the float dequant
 * (ScalarQ6_KMatmulKernel / ggml dequantize_row_q6_K) but without folding in
 * `d`/`scale`: those are applied per scale-group in the int dot, so the inner
 * product stays integer. Two 128-element halves, each with two 16-element scale
 * groups carrying four strided sub-codes (q1..q4) at output offsets +0/+32/+64/+96.
 */
static inline void skainet_q6k_unpack_codes(const uint8_t* SKAINET_RESTRICT block,
                                            int8_t* SKAINET_RESTRICT codes) {
    const uint8_t* ql0 = block + Q6K_QL_OFFSET;
    const uint8_t* qh0 = block + Q6K_QH_OFFSET;

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql = ql0 + half * 64;
        const uint8_t* qh = qh0 + half * 32;
        int8_t* out = codes + half * 128;
        for (int is = 0; is < 2; ++is) {
            const int l_start = is * 16;
            for (int l = l_start; l < l_start + 16; ++l) {
                const int q_l0  = ql[l];
                const int q_l32 = ql[l + 32];
                const int q_h   = qh[l];
                out[l +  0] = (int8_t)(((q_l0  & 0x0F) | ((q_h        & 0x03) << 4)) - 32);
                out[l + 32] = (int8_t)(((q_l32 & 0x0F) | (((q_h >> 2) & 0x03) << 4)) - 32);
                out[l + 64] = (int8_t)(((q_l0  >> 4)   | (((q_h >> 4) & 0x03) << 4)) - 32);
                out[l + 96] = (int8_t)(((q_l32 >> 4)   | (((q_h >> 6) & 0x03) << 4)) - 32);
            }
        }
    }
}

/*
 * Weighted integer dot of one Q6_K block: Σ_g sc[g] · Σ_{i∈g} q8[i]·codes[i],
 * over the 16 scale-groups (each a 16-element contiguous run in natural order).
 * Run `r` for (half,k,is) starts at half*128 + 32*k + is*16 and uses signed
 * scale sc[half*8 + is + 2*k]. On AArch64 with dotprod each 16-element dot is a
 * single vdotq_s32; otherwise a scalar fallback (auto-vectorizes under -O3).
 */
#if defined(SKAINET_HAVE_DOTPROD) || defined(SKAINET_DOTPROD_DISPATCH)
SKAINET_DOTPROD_TARGET
static int64_t skainet_q6k_weighted_dot_dp(const int8_t* SKAINET_RESTRICT q8,
                                           const int8_t* SKAINET_RESTRICT codes,
                                           const int8_t* SKAINET_RESTRICT sc) {
    int64_t sum = 0;
    for (int half = 0; half < 2; ++half) {
        for (int k = 0; k < 4; ++k) {
            for (int is = 0; is < 2; ++is) {
                const int start = half * 128 + 32 * k + is * 16;
                const int gs = half * 8 + is + 2 * k;
                const int32x4_t acc = vdotq_s32(vdupq_n_s32(0),
                    vld1q_s8(codes + start), vld1q_s8(q8 + start));
                sum += (int64_t) sc[gs] * vaddvq_s32(acc);
            }
        }
    }
    return sum;
}
#endif

#if !defined(SKAINET_HAVE_DOTPROD) || defined(SKAINET_DOTPROD_DISPATCH)
static int64_t skainet_q6k_weighted_dot_generic(const int8_t* SKAINET_RESTRICT q8,
                                                const int8_t* SKAINET_RESTRICT codes,
                                                const int8_t* SKAINET_RESTRICT sc) {
    int64_t sum = 0;
    for (int half = 0; half < 2; ++half) {
        for (int k = 0; k < 4; ++k) {
            for (int is = 0; is < 2; ++is) {
                const int start = half * 128 + 32 * k + is * 16;
                const int gs = half * 8 + is + 2 * k;
                int32_t dot = 0;
                for (int j = 0; j < 16; ++j) dot += (int) q8[start + j] * (int) codes[start + j];
                sum += (int64_t) sc[gs] * dot;
            }
        }
    }
    return sum;
}
#endif

/*
 * Native Q6_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q6KMatmulKernel SPI contract. A single
 * input row times an `outputDim x inputDim` Q6_K-packed weight tensor
 * laid out (blockIdx * outputDim + o) * 210 bytes.
 *
 * Fused int8 dot path (ggml-style, mirrors q4k_matmul.c): the input row is
 * quantized to Q8 ONCE per 256-block (reused across all output rows), the 6-bit
 * weight is unpacked to centered int8 codes, and each scale-group is an int8
 * dot (vdotq_s32 on dotprod targets) — no 256-float scratch, no per-element
 * float multiply. acc = d · d_in · Σ_g sc[g]·Σ_{i∈g} q8[i]·codes[i].
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

#ifdef SKAINET_DOTPROD_DISPATCH
    /* One probe per matmul call; cached in skainet_cpu_has_dotprod. */
    const int use_dp = skainet_cpu_has_dotprod();
#endif

    const int32_t blocks_per_input_dim = input_dim / Q6K_BLOCK_SIZE;
    const float* in_base = input + input_offset;
    float* out_base = output + output_offset;

    /* Pre-quantize the whole input row to Q8 once (reused across all o). */
    int8_t* q8 = (int8_t*) malloc((size_t) input_dim * sizeof(int8_t));
    float* d_in = (float*) malloc((size_t) blocks_per_input_dim * sizeof(float));
    if (q8 == NULL || d_in == NULL) { free(q8); free(d_in); return; }
    for (int32_t b = 0; b < blocks_per_input_dim; ++b) {
        d_in[b] = skainet_q6k_q8_quantize_block(in_base + (size_t) b * Q6K_BLOCK_SIZE,
                                                q8 + (size_t) b * Q6K_BLOCK_SIZE);
    }

    int8_t codes[Q6K_BLOCK_SIZE];

    /*
     * Loop order: block OUTER, output row INNER — see q4k_matmul.c for the
     * rationale. The weight is block-major (blockIdx*output_dim + o)*210, so for
     * a fixed block consecutive `o` are 210 bytes apart: the weight bytes are
     * read sequentially (cache/prefetch friendly) instead of striding
     * output_dim*210 per step. out_base[o] accumulates across blocks; the order
     * over blocks is unchanged.
     */
    for (int32_t o = 0; o < output_dim; ++o) out_base[o] = 0.0f;

    for (int32_t block_idx = 0; block_idx < blocks_per_input_dim; ++block_idx) {
        const int8_t* q8_block = q8 + (size_t) block_idx * Q6K_BLOCK_SIZE;
        const float di = d_in[block_idx];
        const uint8_t* block = weight + weight_byte_offset
            + (size_t)(block_idx * output_dim) * Q6K_BYTES_PER_BLOCK;

        for (int32_t o = 0; o < output_dim; ++o, block += Q6K_BYTES_PER_BLOCK) {
            const uint16_t d_bits = (uint16_t) block[Q6K_D_OFFSET]
                | ((uint16_t) block[Q6K_D_OFFSET + 1] << 8);
            const float d = skainet_q6k_half_to_float(d_bits);
            const int8_t* sc = (const int8_t*)(block + Q6K_SCALES_OFFSET);

            skainet_q6k_unpack_codes(block, codes);
#if defined(SKAINET_HAVE_DOTPROD)
            const int64_t wdot = skainet_q6k_weighted_dot_dp(q8_block, codes, sc);
#elif defined(SKAINET_DOTPROD_DISPATCH)
            const int64_t wdot = use_dp
                ? skainet_q6k_weighted_dot_dp(q8_block, codes, sc)
                : skainet_q6k_weighted_dot_generic(q8_block, codes, sc);
#else
            const int64_t wdot = skainet_q6k_weighted_dot_generic(q8_block, codes, sc);
#endif

            out_base[o] += d * di * (float) wdot;
        }
    }

    free(q8);
    free(d_in);
}
