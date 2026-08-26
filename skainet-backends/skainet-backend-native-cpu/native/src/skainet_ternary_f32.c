/*
 * Adapter over the vendored NeoGPU ternary LUT kernel
 * (src/vendor/neogpu/hs_ml_ternary_neon.c, MIT, vendored verbatim — see the
 * vendor README). Exposes the two useful symbols under the skainet_kernels
 * ABI and closes the vendored file's non-atomic LUT-init guard with a
 * pthread_once warm-up, so first use is race-free under any threading.
 *
 * The vendored file needs pthreads and GNU/Clang builtins, so it is only in
 * the build where SKAINET_HAVE_NEOGPU_TERNARY is defined (non-MSVC). The
 * fallback branch below is a portable scalar mirror with identical semantics,
 * including the byte-code-3 → +2.0 decode (loaders reject code 3 at import;
 * the kernel contract is "garbage in, garbage out", never a crash).
 */

#include "skainet_kernels.h"

#include <string.h>

#ifdef SKAINET_HAVE_NEOGPU_TERNARY

#include <pthread.h>

/* The vendored file ships no header; these mirror its public definitions. */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                   const uint8_t *W, uint32_t N, uint32_t K);
extern void hs_ml_lmhead_stage1(float *out, const float *in,
                                const uint8_t *P0, const uint8_t *P1,
                                const uint8_t *P2, const uint8_t *P3,
                                const uint16_t *row_scale,
                                uint32_t N, uint32_t K);

static pthread_once_t skainet_ternary_lut_once = PTHREAD_ONCE_INIT;

static void skainet_ternary_lut_warmup(void) {
    /* Any call builds the 256-entry LUT; N=1 stays on the calling thread
     * (far below the vendored THREAD_THRESHOLD of 512). */
    static const float in[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    static const uint8_t w[1] = { 0 };
    float out;
    hs_ml_ternary_f32_proj(&out, in, w, 1u, 4u);
}

void skainet_ternary_f32_gemv(
    const float* input, int32_t input_offset,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    pthread_once(&skainet_ternary_lut_once, skainet_ternary_lut_warmup);
    /* input_dim == 0 falls through: the vendored kernel writes 0.0f per row. */
    hs_ml_ternary_f32_proj(
        output + output_offset,
        input + input_offset,
        weight + weight_byte_offset,
        (uint32_t)output_dim,
        (uint32_t)input_dim);
}

void skainet_ternary_lmhead_stage1(
    const float* input, int32_t input_offset,
    const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
    const uint16_t* row_scale, int32_t row_scale_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    pthread_once(&skainet_ternary_lut_once, skainet_ternary_lut_warmup);
    const uint8_t* base = planes + planes_byte_offset;
    hs_ml_lmhead_stage1(
        output + output_offset,
        input + input_offset,
        base,
        base + (size_t)plane_stride_bytes,
        base + (size_t)plane_stride_bytes * 2,
        base + (size_t)plane_stride_bytes * 3,
        row_scale + row_scale_offset,
        (uint32_t)output_dim,
        (uint32_t)input_dim);
}

#else /* !SKAINET_HAVE_NEOGPU_TERNARY — portable scalar mirror (MSVC etc.) */

/* decode(byte, lane) = ((byte >> (lane*2)) & 3) - 1, exactly the vendored LUT
 * (so byte code 3 yields +2, matching TernaryCodec.decodeBitNet). */
static float skainet_ternary_decode(uint8_t b, int lane) {
    return (float)((int)((b >> (lane * 2)) & 3) - 1);
}

void skainet_ternary_f32_gemv(
    const float* input, int32_t input_offset,
    const uint8_t* weight, int32_t weight_byte_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    const float* in = input + input_offset;
    const uint8_t* w = weight + weight_byte_offset;
    const int32_t row_bytes = input_dim / 4;
    for (int32_t n = 0; n < output_dim; n++) {
        const uint8_t* wrow = w + (size_t)n * row_bytes;
        float acc = 0.0f;
        for (int32_t bi = 0; bi < row_bytes; bi++) {
            const uint8_t b = wrow[bi];
            acc += skainet_ternary_decode(b, 0) * in[bi * 4    ];
            acc += skainet_ternary_decode(b, 1) * in[bi * 4 + 1];
            acc += skainet_ternary_decode(b, 2) * in[bi * 4 + 2];
            acc += skainet_ternary_decode(b, 3) * in[bi * 4 + 3];
        }
        output[output_offset + n] = acc;
    }
}

void skainet_ternary_lmhead_stage1(
    const float* input, int32_t input_offset,
    const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
    const uint16_t* row_scale, int32_t row_scale_offset,
    int32_t input_dim, int32_t output_dim,
    float* output, int32_t output_offset)
{
    if (output_dim <= 0) return;
    const float* in = input + input_offset;
    const uint8_t* base = planes + planes_byte_offset;
    const uint16_t* rs = row_scale + row_scale_offset;
    const int32_t row_bytes = input_dim / 4;
    const float pw[4] = { 1.0f, 1.0f / 3.0f, 1.0f / 9.0f, 1.0f / 27.0f };
    for (int32_t n = 0; n < output_dim; n++) {
        float acc = 0.0f;
        for (int32_t bi = 0; bi < row_bytes; bi++) {
            for (int lane = 0; lane < 4; lane++) {
                float u = 0.0f;
                for (int p = 0; p < 4; p++) {
                    const uint8_t b =
                        base[(size_t)plane_stride_bytes * p + (size_t)n * row_bytes + bi];
                    u += skainet_ternary_decode(b, lane) * pw[p];
                }
                acc += u * in[bi * 4 + lane];
            }
        }
        /* FP16 row scale decode, same bit arithmetic as the vendored file
         * (sign ignored — encoders store max|row| >= 0). */
        const uint32_t h16 = rs[n];
        const uint32_t eu = (h16 >> 10) & 0x1Fu;
        const uint32_t mu = h16 & 0x03FFu;
        const uint32_t fu = ((eu + 112u) << 23) | (mu << 13);
        float rsc;
        memcpy(&rsc, &fu, 4);
        output[output_offset + n] = acc * rsc;
    }
}

#endif /* SKAINET_HAVE_NEOGPU_TERNARY */
