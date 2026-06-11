#include "skainet_kernels.h"
#include "skainet_simd.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Native row-major SGEMM matching the
 * sk.ainet.backend.api.kernel.Fp32MatmulKernel SPI:
 *
 *   C(m, n) = A(m, k) * B(k, n)
 *
 * Strides are in floats (not bytes); for a contiguous parent matrix
 * `aStride == k`, `bStride == n`, `cStride == n`. Sub-block scenarios
 * pass larger strides and the corresponding offsets.
 *
 * Iteration order is i-p-j (outer product into rows of C). The inner
 * loop is `c[j] += a_ip * b[j]` over a contiguous run of `n` floats
 * for both b's row and c's row — auto-vectorizes cleanly under
 * -O3 -ffast-math into vfmadd231ps / fmla.
 *
 * Caller contract:
 *  - C is FULLY OVERWRITTEN in the m×n block (zero-then-accumulate).
 *  - k == 0 zeros the m×n block.
 *  - m == 0 || n == 0 is a no-op.
 *  - Negative m / n / k are caller errors; the Kotlin wrapper rejects
 *    them. The C kernel still treats negatives as no-op (via the
 *    `<=` loop bounds) defensively.
 */
SKAINET_API void skainet_fp32_matmul(
    const float* SKAINET_RESTRICT a, int32_t a_offset, int32_t a_stride,
    const float* SKAINET_RESTRICT b, int32_t b_offset, int32_t b_stride,
    float* SKAINET_RESTRICT c, int32_t c_offset, int32_t c_stride,
    int32_t m, int32_t n, int32_t k
) {
    if (m <= 0 || n <= 0) return;

    /* Zero the output block. Required by the SPI contract for k == 0
     * AND prerequisite for the i-p-j accumulator pattern below. */
    for (int32_t i = 0; i < m; ++i) {
        float* SKAINET_RESTRICT c_row = c + c_offset + (size_t) i * c_stride;
        for (int32_t j = 0; j < n; ++j) {
            c_row[j] = 0.0f;
        }
    }
    if (k <= 0) return;

    /* Outer-product accumulator: streams two contiguous rows on the
     * inner loop (b's row and c's row), broadcasts a single A scalar.
     * The compiler emits vfmadd231ps with a vbroadcastss for a_ip. */
    for (int32_t i = 0; i < m; ++i) {
        const float* SKAINET_RESTRICT a_row = a + a_offset + (size_t) i * a_stride;
        float* SKAINET_RESTRICT c_row = c + c_offset + (size_t) i * c_stride;
        for (int32_t p = 0; p < k; ++p) {
            const float a_ip = a_row[p];
            const float* SKAINET_RESTRICT b_row = b + b_offset + (size_t) p * b_stride;
#ifdef SKAINET_HAVE_NEON
            const float32x4_t va = vdupq_n_f32(a_ip);
            int32_t j = 0;
            for (; j + 4 <= n; j += 4) {
                float32x4_t cv = vld1q_f32(c_row + j);
                cv = vfmaq_f32(cv, va, vld1q_f32(b_row + j));
                vst1q_f32(c_row + j, cv);
            }
            for (; j < n; ++j) {
                c_row[j] += a_ip * b_row[j];
            }
#else
            for (int32_t j = 0; j < n; ++j) {
                c_row[j] += a_ip * b_row[j];
            }
#endif
        }
    }
}
