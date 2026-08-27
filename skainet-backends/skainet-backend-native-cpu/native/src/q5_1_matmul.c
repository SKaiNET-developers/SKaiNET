#include <stdlib.h>
#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Native FP32 × Q5_1 matrix-vector matmul matching the
 * sk.ainet.backend.api.kernel.Q5_1MatmulKernel SPI.
 *
 * Block layout (canonical ggml Q5_1, 32 elements, 24 bytes):
 *   - bytes 0..1  : FP16 little-endian scale `d`
 *   - bytes 2..3  : FP16 little-endian min `m`
 *   - bytes 4..7  : `qh` — 32-bit little-endian high-bit plane, bit j is
 *     the fifth bit of element j
 *   - bytes 8..23 : 16 bytes packing 32 4-bit codes in the *split*
 *     layout — low nibbles decode elements 0..15, high nibbles decode
 *     elements 16..31.
 *
 * Per-block packed weight layout:
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 24
 *
 * Dequant per element: `d * code + m` with
 * `code = nibble | (fifth_bit << 4)` (unsigned, 0..31, affine — no
 * re-centring). The kernel uses the algebraic split
 *
 *   sum_j x_j * (d * code_j + m) = d * dot(x, code) + m * sum(x)
 *
 * so the inner loop accumulates the unsigned code dot product and the
 * per-block input sum is hoisted OUT of the output-row loop (it only
 * depends on the activations, not on `o`). `d`/`m` fold once per block.
 *
 * Loop order: block OUTER, output row INNER — see q8_0_matmul.c for the
 * rationale (sequential weight reads; per-row accumulation order stays
 * ascending-block, so results don't depend on output_dim).
 *
 * NEON path (SKAINET_HAVE_NEON): plain NEON only — no dotprod/i8mm
 * requirement, so the body runs on every AArch64 core. Identical
 * high-bit expansion + unsigned widen + vfmaq_f32 structure as
 * q5_0_matmul.c; only the per-block fold differs (`+ m*sum` instead of
 * `- 16d*sum`).
 */

/* Portable FP16 → FP32 conversion. Matches the Kotlin `decodeHalf`
 * algorithm bit-for-bit. */
static inline float skainet_q5_1_fp16_to_fp32(uint16_t h) {
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

#define Q51_BLOCK_SIZE       32
#define Q51_BYTES_PER_BLOCK  24

/*
 * One block's contribution to out[o] — the loop body of the original kernel,
 * shared by the feed-order and row-major entries so both orders (and any row
 * partition, #1195) stay bit-identical per output row (#1192).
 */
static inline float q5_1_block_term(
    const uint8_t* SKAINET_RESTRICT block,
    const float* SKAINET_RESTRICT input_block, float input_sum
) {
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            uint16_t m_bits = (uint16_t) block[2] | ((uint16_t) block[3] << 8);
            float d = skainet_q5_1_fp16_to_fp32(d_bits);
            float m = skainet_q5_1_fp16_to_fp32(m_bits);
            const uint8_t* SKAINET_RESTRICT qh = block + 4;
            const uint8_t* SKAINET_RESTRICT qs = block + 8;
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
                return d * code_dot + m * input_sum;
}

/* Everything a row-range worker needs; read-only during the parallel section. */
typedef struct {
    const uint8_t* weight_base;   /* weight + weight_byte_offset */
    const float* in_base;         /* input + input_offset */
    float* out_base;
    const float* in_sums;         /* per-block activation sums (hoisted; see term) */
    int32_t blocks_per_input_dim;
    int32_t output_dim;
} q5_1_ctx;

/* Feed-order rows [o_start, o_end): block OUTER, row INNER — see q4k_matmul.c. */
static void q5_1_rows_feed(void* vctx, int32_t o_start, int32_t o_end) {
    const q5_1_ctx* c = (const q5_1_ctx*) vctx;
    for (int32_t o = o_start; o < o_end; ++o) c->out_base[o] = 0.0f;
    for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim; ++block_idx) {
        const float* input_block = c->in_base + (size_t) block_idx * Q51_BLOCK_SIZE;
        const uint8_t* block = c->weight_base
            + ((size_t) block_idx * c->output_dim + o_start) * Q51_BYTES_PER_BLOCK;
        for (int32_t o = o_start; o < o_end; ++o, block += Q51_BYTES_PER_BLOCK) {
            c->out_base[o] += q5_1_block_term(block, input_block, c->in_sums[block_idx]);
        }
    }
}

/* Row-major rows (#1192): canonical GGUF file order, (o·bpr + b)·24 — mmap-fed as-is. */
static void q5_1_rows_rm(void* vctx, int32_t o_start, int32_t o_end) {
    const q5_1_ctx* c = (const q5_1_ctx*) vctx;
    const uint8_t* block = c->weight_base
        + (size_t) o_start * c->blocks_per_input_dim * Q51_BYTES_PER_BLOCK;
    for (int32_t o = o_start; o < o_end; ++o) {
        float acc_row = 0.0f;
        for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim;
             ++block_idx, block += Q51_BYTES_PER_BLOCK) {
            acc_row += q5_1_block_term(block, c->in_base + (size_t) block_idx * Q51_BLOCK_SIZE, c->in_sums[block_idx]);
        }
        c->out_base[o] = acc_row;
    }
}

/* Shared entry — guards, ctx fill, threaded run (skainet_row_threads.h, #1195). */
static void q5_1_matmul_run(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset,
    skainet_row_range_fn worker
) {
    if (output_dim <= 0) return;
    if (input_dim <= 0) {
        for (int32_t o = 0; o < output_dim; ++o) output[output_offset + o] = 0.0f;
        return;
    }
    q5_1_ctx ctx;
    ctx.weight_base = weight + weight_byte_offset;
    ctx.in_base = input + input_offset;
    ctx.out_base = output + output_offset;
    ctx.blocks_per_input_dim = input_dim / Q51_BLOCK_SIZE;
    ctx.output_dim = output_dim;

    float* in_sums = (float*) malloc((size_t) ctx.blocks_per_input_dim * sizeof(float));
    if (in_sums == NULL) return;
    for (int32_t b = 0; b < ctx.blocks_per_input_dim; ++b) {
        const float* ib = ctx.in_base + (size_t) b * Q51_BLOCK_SIZE;
        float sum = 0.0f;
        for (int32_t k = 0; k < Q51_BLOCK_SIZE; ++k) sum += ib[k];
        in_sums[b] = sum;
    }
    ctx.in_sums = in_sums;
    skainet_run_rows(worker, &ctx, output_dim);
    free(in_sums);
}

SKAINET_API void skainet_q5_1_matmul(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    q5_1_matmul_run(input, input_offset, weight, weight_byte_offset,
                     input_dim, output_dim, output, output_offset,
                     q5_1_rows_feed);
}

/*
 * Row-major variant (#1192): canonical GGUF file order — see q5_1_rows_rm.
 * Numerically equivalent to the feed-order kernel (same per-row block order; -ffast-math FMA/reassociation may differ at ULP scale); threads over rows >= 512 (#1195).
 */
SKAINET_API void skainet_q5_1_matmul_rm(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    q5_1_matmul_run(input, input_offset, weight, weight_byte_offset,
                     input_dim, output_dim, output, output_offset,
                     q5_1_rows_rm);
}
