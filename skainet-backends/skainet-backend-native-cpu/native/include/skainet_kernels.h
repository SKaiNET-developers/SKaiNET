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

#ifdef __cplusplus
}
#endif

#endif /* SKAINET_KERNELS_H */
