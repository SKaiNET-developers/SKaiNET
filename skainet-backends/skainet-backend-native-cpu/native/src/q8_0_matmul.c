#include "skainet_kernels.h"
#include "skainet_simd.h"
#include "skainet_row_threads.h"

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

#define Q80_BLOCK_SIZE       32
#define Q80_BYTES_PER_BLOCK  34

/*
 * One block's contribution to out[o]: `d * Σ_k input[k] * code[k]`.
 * Activations are FP32, so int8 codes widen to float and FMA (int8 dotprod
 * would need int8 activations). Shared by the feed-order and row-major
 * entries, so both orders (and any row partition, #1195) stay bit-identical
 * per output row.
 */
static inline float skainet_q8_0_block_term(
    const uint8_t* SKAINET_RESTRICT block,
    const float* SKAINET_RESTRICT input_block
) {
    const uint16_t d_bits = (uint16_t) block[0] | ((uint16_t) block[1] << 8);
    const float d = skainet_fp16_to_fp32(d_bits);
    const int8_t* SKAINET_RESTRICT codes = (const int8_t*) (block + 2);
    float block_sum = 0.0f;
#ifdef SKAINET_HAVE_NEON
    float32x4_t accv = vdupq_n_f32(0.0f);
    for (int32_t k = 0; k < Q80_BLOCK_SIZE; k += 16) {
        const int8x16_t c8 = vld1q_s8(codes + k);
        const int16x8_t lo16 = vmovl_s8(vget_low_s8(c8));
        const int16x8_t hi16 = vmovl_s8(vget_high_s8(c8));
        const float32x4_t cf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
        const float32x4_t cf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
        const float32x4_t cf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
        const float32x4_t cf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
        accv = vfmaq_f32(accv, vld1q_f32(input_block + k),      cf0);
        accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 4),  cf1);
        accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 8),  cf2);
        accv = vfmaq_f32(accv, vld1q_f32(input_block + k + 12), cf3);
    }
    block_sum = skainet_neon_hadd_f32(accv);
#else
    for (int32_t k = 0; k < Q80_BLOCK_SIZE; ++k) {
        block_sum += input_block[k] * (float) codes[k];
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
} skainet_q8_0_ctx;

/*
 * Feed-order rows [o_start, o_end): block OUTER, row INNER — see q4k_matmul.c
 * for the cache rationale. out[o] accumulates across blocks in unchanged
 * order under any partition.
 */
static void skainet_q8_0_rows_feed(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_q8_0_ctx* c = (const skainet_q8_0_ctx*) vctx;
    for (int32_t o = o_start; o < o_end; ++o) c->out_base[o] = 0.0f;
    for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim; ++block_idx) {
        const float* input_block = c->in_base + (size_t) block_idx * Q80_BLOCK_SIZE;
        const uint8_t* block = c->weight_base
            + ((size_t) block_idx * c->output_dim + o_start) * Q80_BYTES_PER_BLOCK;
        for (int32_t o = o_start; o < o_end; ++o, block += Q80_BYTES_PER_BLOCK) {
            c->out_base[o] += skainet_q8_0_block_term(block, input_block);
        }
    }
}

/*
 * Row-major rows [o_start, o_end) (#1189/#1192): canonical GGUF file order —
 * (o * blocks_per_row + b) * 34 — an mmap'd tensor is fed as-is, no relayout
 * copy; each row's blocks are contiguous on disk. Per-row accumulation order
 * matches the feed-order worker's.
 */
static void skainet_q8_0_rows_rm(void* vctx, int32_t o_start, int32_t o_end) {
    const skainet_q8_0_ctx* c = (const skainet_q8_0_ctx*) vctx;
    const uint8_t* block = c->weight_base
        + (size_t) o_start * c->blocks_per_input_dim * Q80_BYTES_PER_BLOCK;
    for (int32_t o = o_start; o < o_end; ++o) {
        float acc = 0.0f;
        for (int32_t block_idx = 0; block_idx < c->blocks_per_input_dim;
             ++block_idx, block += Q80_BYTES_PER_BLOCK) {
            acc += skainet_q8_0_block_term(block, c->in_base + (size_t) block_idx * Q80_BLOCK_SIZE);
        }
        c->out_base[o] = acc;
    }
}

/* Shared entry — guards, ctx fill, threaded run (skainet_row_threads.h, #1195). */
static void skainet_q8_0_matmul_run(
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
    skainet_q8_0_ctx ctx;
    ctx.weight_base = weight + weight_byte_offset;
    ctx.in_base = input + input_offset;
    ctx.out_base = output + output_offset;
    ctx.blocks_per_input_dim = input_dim / Q80_BLOCK_SIZE;
    ctx.output_dim = output_dim;
    skainet_run_rows(worker, &ctx, output_dim);
}

SKAINET_API void skainet_q8_0_matmul(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    skainet_q8_0_matmul_run(input, input_offset, weight, weight_byte_offset,
                            input_dim, output_dim, output, output_offset,
                            skainet_q8_0_rows_feed);
}

/*
 * Row-major variant (#1192): the weight stays in canonical GGUF file order —
 * see skainet_q8_0_rows_rm. Numerically equivalent to the feed-order kernel (same per-row block order; -ffast-math FMA/reassociation may differ at ULP scale); threads
 * over output rows when outputDim >= 512 (#1195).
 */
SKAINET_API void skainet_q8_0_matmul_rm(
    const float* SKAINET_RESTRICT input, int32_t input_offset,
    const uint8_t* SKAINET_RESTRICT weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* SKAINET_RESTRICT output, int32_t output_offset
) {
    skainet_q8_0_matmul_run(input, input_offset, weight, weight_byte_offset,
                            input_dim, output_dim, output, output_offset,
                            skainet_q8_0_rows_rm);
}
