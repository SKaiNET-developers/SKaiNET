#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_cpu_features.h"
#include "skainet_row_threads.h"

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
 * One block's contribution to out[o]:
 *   d_in[b] * ( d * Σ_s scaleIdx[s]*intDot[s] - dMin * Σ_s minIdx[s]*intSum[s] )
 * where intDot[s] = Σ q8[i]*code[i] and intSum[s] = Σ q8[i] over the sub-block.
 * On AArch64 with dotprod (asimddp) the inner dot uses vdotq_s32 (16 int8 MACs
 * per instruction); otherwise a scalar integer fallback (auto-vectorized).
 * Shared by the feed-order and row-major entries — identical math, so the two
 * orders (and any row partition, #1195) stay bit-identical per output row.
 */
static inline float skainet_q4k_block_term(
    const uint8_t* SKAINET_RESTRICT block,
    const int8_t* SKAINET_RESTRICT q8_block,
    float di,
    int use_dp
) {
    int scale_idx[Q4K_SUB_BLOCKS];
    int min_idx[Q4K_SUB_BLOCKS];

    const uint16_t d_bits     = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
    const uint16_t d_min_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
    const float d     = skainet_half_to_float(d_bits);
    const float d_min = skainet_half_to_float(d_min_bits);

    skainet_q4k_decode_scales(block + 4, scale_idx, min_idx);

    const uint8_t* qs = block + 16;

    int64_t block_scale_dot = 0;
    int64_t block_min_sum = 0;

#if defined(SKAINET_HAVE_DOTPROD)
    (void) use_dp;
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
    (void) use_dp;
    skainet_q4k_block_dot_generic(qs, q8_block, scale_idx, min_idx,
                                  &block_scale_dot, &block_min_sum);
#endif

    return di * (d * (float) block_scale_dot - d_min * (float) block_min_sum);
}

/* Everything a row-range worker needs; read-only during the parallel section. */
typedef struct {
    const uint8_t* weight_base;   /* weight + weight_byte_offset */
    const int8_t* q8;
    const float* d_in;
    float* out_base;
    int32_t blocks_per_input_dim;
    int32_t output_dim;
    int use_dp;                   /* meaningful only under SKAINET_DOTPROD_DISPATCH */
} skainet_q4k_ctx;

/*
 * Feed-order rows [o_start, o_end). Loop order: block OUTER, output row INNER.
 * The weight is packed block-major — (blockIdx * output_dim + o) * 144 — so for
 * a fixed block this range's rows are one contiguous 144·(o_end−o_start) byte
 * run: reads stay sequential (prefetch- and cache-line-friendly; the o-outer
 * order would stride output_dim*144 per step, a cold miss per read on an
 * in-order A55). out[o] accumulates across blocks in unchanged order, so the
 * result is numerically identical to the o-outer form and to any partition.
 */
static void skainet_q4k_rows_feed(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_q4k_ctx* c = (const skainet_q4k_ctx*) vctx;
    for (int32_t o = o_start; o < o_end; ++o) c->out_base[o] = 0.0f;
    for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim; ++block_idx) {
        const int8_t* q8_block = c->q8 + (size_t) block_idx * Q4K_BLOCK_SIZE;
        const float di = c->d_in[block_idx];
        const uint8_t* block = c->weight_base
            + ((size_t) block_idx * c->output_dim + o_start) * Q4K_BYTES_PER_BLOCK;
        for (int32_t o = o_start; o < o_end; ++o, block += Q4K_BYTES_PER_BLOCK) {
            c->out_base[o] += skainet_q4k_block_term(block, q8_block, di, c->use_dp);
        }
    }
}

/*
 * Row-major rows [o_start, o_end) (#1189): canonical GGUF file order —
 * (o * blocks_per_row + b) * 144 — so an mmap'd tensor is fed as-is, no
 * relayout copy. o OUTER: each row's blocks are contiguous on disk, reads stay
 * strictly sequential; the Q8 activation (input_dim bytes) stays hot across
 * rows. Per-row accumulation order matches the feed-order worker's.
 */
static void skainet_q4k_rows_rm(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_q4k_ctx* c = (const skainet_q4k_ctx*) vctx;
    const uint8_t* block = c->weight_base
        + (size_t) o_start * c->blocks_per_input_dim * Q4K_BYTES_PER_BLOCK;
    for (int32_t o = o_start; o < o_end; ++o) {
        float acc = 0.0f;
        for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim;
             ++block_idx, block += Q4K_BYTES_PER_BLOCK) {
            acc += skainet_q4k_block_term(block, c->q8 + (size_t) block_idx * Q4K_BLOCK_SIZE,
                                          c->d_in[block_idx], c->use_dp);
        }
        c->out_base[o] = acc;
    }
}

/*
 * Shared entry: quantize the input row to Q8 once (reused across all rows and
 * all threads — read-only after this point), then run the worker over the
 * output rows, threaded per skainet_row_threads.h (#1195: ≥512 rows → up to 4
 * pthreads; below that, or on MSVC, the calling thread does all rows).
 */
static void skainet_q4k_matmul_run(
    const float* SKAINET_RESTRICT input,
    int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* SKAINET_RESTRICT output,
    int32_t output_offset,
    skainet_row_range_fn worker
) {
    if (output_dim <= 0 || input_dim <= 0) return;

    const int32_t blocks_per_input_dim = input_dim / Q4K_BLOCK_SIZE;
    const float* in_base = input + input_offset;

    int8_t* q8 = (int8_t*) malloc((size_t) input_dim * sizeof(int8_t));
    float* d_in = (float*) malloc((size_t) blocks_per_input_dim * sizeof(float));
    if (q8 == NULL || d_in == NULL) { free(q8); free(d_in); return; }
    for (int32_t b = 0; b < blocks_per_input_dim; ++b) {
        d_in[b] = skainet_q8_quantize_block(in_base + (size_t) b * Q4K_BLOCK_SIZE,
                                            q8 + (size_t) b * Q4K_BLOCK_SIZE);
    }

    skainet_q4k_ctx ctx;
    ctx.weight_base = weight + weight_byte_offset;
    ctx.q8 = q8;
    ctx.d_in = d_in;
    ctx.out_base = output + output_offset;
    ctx.blocks_per_input_dim = blocks_per_input_dim;
    ctx.output_dim = output_dim;
#ifdef SKAINET_DOTPROD_DISPATCH
    /* One probe per matmul call; cached in skainet_cpu_has_dotprod. */
    ctx.use_dp = skainet_cpu_has_dotprod();
#else
    ctx.use_dp = 0;
#endif

    skainet_run_rows(worker, &ctx, output_dim);

    free(q8);
    free(d_in);
}

/*
 * Native Q4_K matrix-vector multiply matching the
 * sk.ainet.backend.api.kernel.Q4KMatmulKernel SPI contract. Single input row
 * times an `outputDim x inputDim` Q4_K-packed weight laid out
 * (blockIdx * outputDim + o) * 144 bytes. Threads over output rows when
 * outputDim >= 512 (#1195); parity-checked against Panama.
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
    skainet_q4k_matmul_run(input, input_offset, weight, weight_byte_offset,
                           input_dim, output_dim, output, output_offset,
                           skainet_q4k_rows_feed);
}

/*
 * Row-major variant (#1189): the weight stays in canonical GGUF file order —
 * (o * blocks_per_row + b) * 144 — see skainet_q4k_rows_rm. Bit-identical to
 * the feed-order kernel; threads over output rows when outputDim >= 512.
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
    skainet_q4k_matmul_run(input, input_offset, weight, weight_byte_offset,
                           input_dim, output_dim, output, output_offset,
                           skainet_q4k_rows_rm);
}
