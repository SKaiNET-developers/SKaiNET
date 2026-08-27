#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

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
 *
 * Loop order: block OUTER, output row INNER — see q8_0_matmul.c for the
 * rationale. The weight is block-major, so for a fixed block consecutive
 * `o` are 18 bytes apart: weight bytes are read sequentially instead of
 * striding output_dim*18 per step. Per output row the blocks still
 * accumulate in ascending order, so results are numerically identical to
 * the previous row-outer form.
 *
 * NEON path (SKAINET_HAVE_NEON): unpack the 16 code bytes into two
 * int8x16 nibble vectors (vand / vshr), re-centre with vsubq_s8(.., 8),
 * widen to f32 lanes and FMA against the activation lanes — the same
 * widen+vfmaq_f32 structure as q8_0_matmul.c, because activations are
 * FP32 (an int8 dotprod would need int8 activations). Plain NEON only,
 * no dotprod/i8mm requirement, so the body runs on every AArch64 core.
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

#define Q40_BLOCK_SIZE       32
#define Q40_BYTES_PER_BLOCK  18

/*
 * One block's contribution to out[o] — the loop body of the original kernel,
 * shared by the feed-order and row-major entries so both orders (and any row
 * partition, #1195) stay bit-identical per output row (#1192).
 */
static inline float q4_0_block_term(
    const uint8_t* SKAINET_RESTRICT block,
    const float* SKAINET_RESTRICT input_block
) {
            uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
            float d = skainet_q4_0_fp16_to_fp32(d_bits);
            const uint8_t* SKAINET_RESTRICT codes = block + 2;
            float block_sum = 0.0f;
#ifdef SKAINET_HAVE_NEON
            /* Split-nibble unpack: low nibbles are elements 0..15, high
             * nibbles are elements 16..31. Re-centre by 8 in the signed
             * int8 domain, then widen to f32 and FMA — same structure as
             * the q8_0 NEON body. */
            const uint8x16_t packed = vld1q_u8(codes);
            const int8x16_t lo8 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F))),
                vdupq_n_s8(8));
            const int8x16_t hi8 = vsubq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)),
                vdupq_n_s8(8));
            const int16x8_t lo16a = vmovl_s8(vget_low_s8(lo8));  /* elems 0..7   */
            const int16x8_t lo16b = vmovl_s8(vget_high_s8(lo8)); /* elems 8..15  */
            const int16x8_t hi16a = vmovl_s8(vget_low_s8(hi8));  /* elems 16..23 */
            const int16x8_t hi16b = vmovl_s8(vget_high_s8(hi8)); /* elems 24..31 */
            float32x4_t accv = vdupq_n_f32(0.0f);
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 0),
                             vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16a))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 4),
                             vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16a))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 8),
                             vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16b))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 12),
                             vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16b))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 16),
                             vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16a))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 20),
                             vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16a))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 24),
                             vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16b))));
            accv = vfmaq_f32(accv, vld1q_f32(input_block + 28),
                             vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16b))));
            block_sum = skainet_neon_hadd_f32(accv);
#else
            for (int32_t k = 0; k < 16; ++k) {
                int32_t lo = (int32_t)(codes[k] & 0x0F) - 8;
                int32_t hi = (int32_t)(codes[k] >> 4) - 8;
                block_sum += input_block[k] * (float) lo;
                block_sum += input_block[k + 16] * (float) hi;
            }
#endif
                return block_sum * d;
}

/* Everything a row-range worker needs; read-only during the parallel section. */
typedef struct {
    const uint8_t* weight_base;   /* weight + weight_byte_offset */
    const float* in_base;         /* input + input_offset */
    float* out_base;
    int32_t blocks_per_input_dim;
    int32_t output_dim;
} q4_0_ctx;

/* Feed-order rows [o_start, o_end): block OUTER, row INNER — see q4k_matmul.c. */
static void q4_0_rows_feed(void* vctx, int32_t o_start, int32_t o_end) {
    const q4_0_ctx* c = (const q4_0_ctx*) vctx;
    for (int32_t o = o_start; o < o_end; ++o) c->out_base[o] = 0.0f;
    for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim; ++block_idx) {
        const float* input_block = c->in_base + (size_t) block_idx * Q40_BLOCK_SIZE;
        const uint8_t* block = c->weight_base
            + ((size_t) block_idx * c->output_dim + o_start) * Q40_BYTES_PER_BLOCK;
        for (int32_t o = o_start; o < o_end; ++o, block += Q40_BYTES_PER_BLOCK) {
            c->out_base[o] += q4_0_block_term(block, input_block);
        }
    }
}

/* Row-major rows (#1192): canonical GGUF file order, (o·bpr + b)·18 — mmap-fed as-is. */
static void q4_0_rows_rm(void* vctx, int32_t o_start, int32_t o_end) {
    const q4_0_ctx* c = (const q4_0_ctx*) vctx;
    const uint8_t* block = c->weight_base
        + (size_t) o_start * c->blocks_per_input_dim * Q40_BYTES_PER_BLOCK;
    for (int32_t o = o_start; o < o_end; ++o) {
        float acc_row = 0.0f;
        for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim;
             ++block_idx, block += Q40_BYTES_PER_BLOCK) {
            acc_row += q4_0_block_term(block, c->in_base + (size_t) block_idx * Q40_BLOCK_SIZE);
        }
        c->out_base[o] = acc_row;
    }
}

/* Shared entry — guards, ctx fill, threaded run (skainet_row_threads.h, #1195). */
static void q4_0_matmul_run(
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
    q4_0_ctx ctx;
    ctx.weight_base = weight + weight_byte_offset;
    ctx.in_base = input + input_offset;
    ctx.out_base = output + output_offset;
    ctx.blocks_per_input_dim = input_dim / Q40_BLOCK_SIZE;
    ctx.output_dim = output_dim;
    skainet_run_rows(worker, &ctx, output_dim);
}

SKAINET_API void skainet_q4_0_matmul(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    q4_0_matmul_run(input, input_offset, weight, weight_byte_offset,
                     input_dim, output_dim, output, output_offset,
                     q4_0_rows_feed);
}

/*
 * Row-major variant (#1192): canonical GGUF file order — see q4_0_rows_rm.
 * Numerically equivalent to the feed-order kernel (same per-row block order; -ffast-math FMA/reassociation may differ at ULP scale); threads over rows >= 512 (#1195).
 */
SKAINET_API void skainet_q4_0_matmul_rm(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    q4_0_matmul_run(input, input_offset, weight, weight_byte_offset,
                     input_dim, output_dim, output, output_offset,
                     q4_0_rows_rm);
}
