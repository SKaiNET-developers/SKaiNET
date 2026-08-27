#ifndef SKAINET_KERNELS_H
#define SKAINET_KERNELS_H

#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#  define SKAINET_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define SKAINET_API __attribute__((visibility("default")))
#else
#  define SKAINET_API
#endif

/* Portable "restrict" qualifier: GNU/Clang accept __restrict__,
 * MSVC accepts __restrict, and the C99 keyword `restrict` is
 * unreliable across compiler modes. */
#if defined(__GNUC__) || defined(__clang__)
#  define SKAINET_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define SKAINET_RESTRICT __restrict
#else
#  define SKAINET_RESTRICT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Trivial smoke kernel proving the FFM downcall pipeline end-to-end.
 *
 *   for (int i = 0; i < length; ++i) output[i] = 2.0f * input[i];
 *
 * The Kotlin caller owns the memory backing `input` and `output`; the
 * kernel must not retain pointers past return.
 */
SKAINET_API void skainet_smoke_double(const float* input, float* output, int32_t length);

/*
 * Q4_K matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q4_K, 256 elements per super-block, 144
 * bytes per block, with packed weights laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 144
 *
 * Caller owns input/weight/output memory; the kernel does not retain
 * pointers past return. input_dim must be a multiple of 256.
 *
 * Threads over output rows (disjoint out[] slices, pthreads, up to 4)
 * when output_dim >= 512 (#1195); single-threaded below that and on
 * MSVC. Results are bit-identical either way — per output row the
 * accumulation order over blocks never changes.
 */
SKAINET_API void skainet_q4k_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q4_K matrix-vector multiply over a ROW-MAJOR (canonical GGUF file order)
 * weight (#1189). Same math and block format as skainet_q4k_matmul; the
 * only difference is the weight addressing:
 *   weight + weight_byte_offset + (o * blocks_per_row + block_idx) * 144
 * i.e. the bytes exactly as they sit in a .gguf file — which is what lets
 * mmap'd weights be fed to this kernel with no relayout copy (each row's
 * blocks are read strictly sequentially).
 *
 * Threads over output rows when output_dim >= 512 (#1195) — see
 * skainet_q4k_matmul; bit-identical to the single-threaded result.
 */
SKAINET_API void skainet_q4k_matmul_rm(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q5_K matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q5_K, 256 elements per super-block, 176
 * bytes per block (2 B d + 2 B dMin + 12 B packed scales + 32 B `qh`
 * high-bit plane + 128 B `qs` low nibbles). Each 5-bit code is
 * `lowNibble | (fifthBit << 4)`. Packed weights laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 176
 *
 * input_dim must be a multiple of 256.
 */
SKAINET_API void skainet_q5k_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q6_K matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q6_K, 256 elements per super-block, 210
 * bytes per block (128 B `ql` low nibbles + 64 B `qh` high-2-bit plane +
 * 16 B int8 `scales` + 2 B `d` FP16). Each 6-bit code is
 * `lowNibble | (highBits << 4)`, dequantized as `d * scale * (code - 32)`
 * (signed, range [-32, 31]; Q6_K has no per-block min). Packed weights
 * laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 210
 *
 * input_dim must be a multiple of 256.
 *
 * Threads over output rows when output_dim >= 512 (#1195) — see
 * skainet_q4k_matmul; bit-identical to the single-threaded result.
 */
SKAINET_API void skainet_q6k_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q6_K matrix-vector multiply over a ROW-MAJOR (canonical GGUF file order)
 * weight (#1189). Same math and block format as skainet_q6k_matmul; the
 * weight is addressed
 *   weight + weight_byte_offset + (o * blocks_per_row + block_idx) * 210
 * — the bytes exactly as they sit in a .gguf file, so an mmap'd weight
 * needs no relayout copy.
 *
 * Threads over output rows when output_dim >= 512 (#1195) — see
 * skainet_q4k_matmul; bit-identical to the single-threaded result.
 */
SKAINET_API void skainet_q6k_matmul_rm(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Row-major FP32 SGEMM:  C(m, n) = A(m, k) * B(k, n).
 *
 * Strides are in floats (not bytes). For a contiguous parent matrix
 * `a_stride == k`, `b_stride == n`, `c_stride == n`. The kernel zeros
 * the m×n output block before accumulating, so callers always get
 * `C = A·B` (not `C += A·B`). `k == 0` zeros the block; `m == 0`
 * or `n == 0` is a no-op.
 */
SKAINET_API void skainet_fp32_matmul(
    const float* a, int32_t a_offset, int32_t a_stride,
    const float* b, int32_t b_offset, int32_t b_stride,
    float* c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
);

/*
 * Row-major FP32 × BF16 matmul: C(m, n) = A(m, k) * B(k, n).
 *
 * A and C are FP32; B is packed BF16 little-endian (2 bytes per element).
 * Strides for A and C are in floats (`a_stride == k` for a contiguous
 * parent). Strides for B are in *bytes* (`b_byte_stride == n * 2` for a
 * contiguous parent). The kernel zeros the m×n output block before
 * accumulating, so callers always get `C = A·B` (not `C += A·B`).
 * `k == 0` zeros the block; `m == 0` or `n == 0` is a no-op.
 *
 * BF16 → FP32 conversion is a bit-shift: `fp32_bits = ((u32) bf16) << 16`.
 */
SKAINET_API void skainet_bf16_matmul(
    const float* a, int32_t a_offset, int32_t a_stride,
    const uint8_t* b, int32_t b_byte_offset, int32_t b_byte_stride,
    float* c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
);

/*
 * Row-major FP32 × FP16 matmul: C(m, n) = A(m, k) * B(k, n).
 *
 * Identical contract to skainet_bf16_matmul, with B packed as IEEE
 * binary16 little-endian (2 bytes per element) instead of BF16.
 *
 * FP16 → FP32 needs exponent rebiasing and gradual-underflow handling
 * rather than BF16's single shift; it is done branch-free so the inner
 * loop still vectorizes. See src/fp16_matmul.c.
 */
SKAINET_API void skainet_fp16_matmul(
    const float* a, int32_t a_offset, int32_t a_stride,
    const uint8_t* b, int32_t b_byte_offset, int32_t b_byte_stride,
    float* c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
);

/*
 * Q8_0 matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q8_0, 32 elements per block, 34 bytes
 * per block (2 B FP16 scale + 32 B int8 codes), with packed weights
 * laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 34
 *
 * input_dim must be a multiple of 32.
 */
SKAINET_API void skainet_q8_0_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q4_0 matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q4_0, 32 elements per block, 18 bytes
 * per block (2 B FP16 scale + 16 B packed 4-bit codes in split layout —
 * low nibbles → elements 0..15, high nibbles → 16..31), with packed
 * weights laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 18
 *
 * Dequant per element: `(code - 8) * d`. input_dim must be a multiple
 * of 32.
 */
SKAINET_API void skainet_q4_0_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q5_0 matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q5_0, 32 elements per block, 22 bytes
 * per block (2 B FP16 scale `d` + 4 B `qh` high-bit plane + 16 B packed
 * 4-bit codes in split layout — low nibbles → elements 0..15, high
 * nibbles → 16..31; bit j of the little-endian 32-bit `qh` is the fifth
 * bit of element j), with packed weights laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 22
 *
 * Dequant per element: `(code - 16) * d` with
 * `code = nibble | (fifth_bit << 4)`. input_dim must be a multiple
 * of 32.
 */
SKAINET_API void skainet_q5_0_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * Q5_1 matrix-vector multiply.
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                dequant(weight[block, o, j])
 *
 * Block layout: canonical ggml Q5_1, 32 elements per block, 24 bytes
 * per block (2 B FP16 scale `d` + 2 B FP16 min `m` + 4 B `qh` high-bit
 * plane + 16 B packed 4-bit codes in split layout; bit j of the
 * little-endian 32-bit `qh` is the fifth bit of element j), with packed
 * weights laid out as
 *   weight + weight_byte_offset + (block_idx * output_dim + o) * 24
 *
 * Dequant per element: `d * code + m` (affine, no re-centring) with
 * `code = nibble | (fifth_bit << 4)`. input_dim must be a multiple
 * of 32.
 */
SKAINET_API void skainet_q5_1_matmul(
    const float* input,
    int32_t input_offset,
    const uint8_t* weight,
    int32_t weight_byte_offset,
    int32_t input_dim,
    int32_t output_dim,
    float* output,
    int32_t output_offset
);

/*
 * `bitnet_gemv`: int8 activations (one absmax scale per token) against ternary
 * TQ2_0 weights — the SKEEP-003 §5.3 kernel, matching
 * sk.ainet.backend.api.kernel.BitNetGemvKernel.
 *
 * A ternary weight is an add, a subtract or nothing; the vector work is the
 * unpacking, after which `sdot` accumulates sixteen products per instruction
 * where the core has ARMv8.2 dotprod.
 *
 * @param activation        int8 codes of one token, `input_dim` of them
 * @param activation_scale  the token's absmax scale (`absmax / 127`)
 * @param weight            canonical TQ2_0 blocks, row-major per output row
 * @param output            `output_dim` floats
 */
SKAINET_API void skainet_bitnet_gemv_tq2_0(
    const int8_t* activation, int32_t activation_offset,
    float activation_scale,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset);

/*
 * Ternary f32 GEMV — exact FP32 activations against sequentially-packed
 * ternary weights (the BitNet b1.58 / `BITNET_B1_58` payload: 4 codes per
 * byte, low bit-pair first, code {0,1,2} → {-1,0,+1}; byte code 3 decodes
 * to +2, loaders reject it at import).
 *
 *   output[output_offset + o] = sum_j input[input_offset + j] *
 *                                decode(weight row o)
 *
 * NO scale is applied — the caller owns the per-tensor scale. Weights are
 * row-major, input_dim/4 bytes per output row, at
 *   weight + weight_byte_offset + o * (input_dim / 4)
 *
 * input_dim must be a multiple of 4. Unlike the int8 `bitnet_gemv` path
 * there is no activation quantization: results are exact. Backed by the
 * vendored NeoGPU LUT kernel (baseline NEON, no dotprod — the fast path for
 * Cortex-A72/Pi-4 class cores); it threads internally with pthreads once
 * output_dim >= 512. A portable scalar build stands in where the vendored
 * file cannot compile (MSVC).
 */
SKAINET_API void skainet_ternary_f32_gemv(
    const float* input, int32_t input_offset,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset);

/*
 * Fused 4-plane ternary lm_head, Stage 1 (NeoGPU multi-plane trit format).
 *
 *   output[output_offset + o] = f16(row_scale[row_scale_offset + o]) *
 *       sum_{p=0}^{3} (1/3^p) * sum_j input[..+j] * decode(plane p, row o)
 *
 * Each plane is a full sequentially-packed ternary matrix (same payload rule
 * as skainet_ternary_f32_gemv); plane p starts at
 *   planes + planes_byte_offset + p * plane_stride_bytes
 * so a single buffer of four concatenated planes uses
 * plane_stride_bytes == output_dim * input_dim / 4. row_scale holds raw
 * little-endian IEEE binary16 bit patterns, one per output row (sign
 * ignored — encoders store max|row| >= 0). input_dim must be a multiple
 * of 4. Threads internally with pthreads at any output_dim.
 */
SKAINET_API void skainet_ternary_lmhead_stage1(
    const float* input, int32_t input_offset,
    const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
    const uint16_t* row_scale, int32_t row_scale_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset);

#ifdef __cplusplus
}
#endif

#endif /* SKAINET_KERNELS_H */
