#include <jni.h>
#include <stdint.h>

#include "skainet_kernels.h"

/*
 * Thin JNI shims over the shared C matmul kernels (skainet_kernels.h).
 * One function per kernel, no logic beyond array pinning + the call.
 *
 * Array strategy: GetPrimitiveArrayCritical for every array. On ART, heap
 * primitive arrays are contiguous, so criticals are zero-copy pins. Rules
 * honored here: no JNI calls between Get and Release, releases in reverse
 * acquisition order, read-only arrays released with JNI_ABORT (no
 * write-back), the output with 0 (write-back + unpin). Kernel calls are
 * millisecond-scale — well within acceptable critical-section length.
 *
 * Method names deliberately contain no underscores (JNI mangles `_` to
 * `_1`, which is easy to get wrong silently).
 */

/* Pin helper: acquire all three arrays, run `CALL`, release in reverse. */
#define SKAINET_JNI_MATMUL_BODY(CALL)                                          \
    jfloat* in = (*env)->GetPrimitiveArrayCritical(env, input, NULL);          \
    jbyte* w = in ? (*env)->GetPrimitiveArrayCritical(env, weight, NULL) : NULL; \
    jfloat* out = w ? (*env)->GetPrimitiveArrayCritical(env, output, NULL) : NULL; \
    if (out) {                                                                 \
        CALL;                                                                  \
    }                                                                          \
    if (out) (*env)->ReleasePrimitiveArrayCritical(env, output, out, 0);       \
    if (w) (*env)->ReleasePrimitiveArrayCritical(env, weight, w, JNI_ABORT);   \
    if (in) (*env)->ReleasePrimitiveArrayCritical(env, input, in, JNI_ABORT);

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_smoke(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jfloatArray output, jint length
) {
    (void) thiz;
    jfloat* in = (*env)->GetPrimitiveArrayCritical(env, input, NULL);
    jfloat* out = in ? (*env)->GetPrimitiveArrayCritical(env, output, NULL) : NULL;
    if (out) {
        skainet_smoke_double(in, out, length);
    }
    if (out) (*env)->ReleasePrimitiveArrayCritical(env, output, out, 0);
    if (in) (*env)->ReleasePrimitiveArrayCritical(env, input, in, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q80Matmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q8_0_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                            inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q40Matmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q4_0_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                            inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q50Matmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5_0_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                            inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q51Matmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5_1_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                            inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q4kMatmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q4k_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                           inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q5kMatmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5k_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                           inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q6kMatmul(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q6k_matmul(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                           inputDim, outputDim, out, outputOffset))
}

/*
 * Direct-buffer row-major matmuls (#1189): the weight arrives as a direct
 * ByteBuffer over mmap'd (or direct-allocated) bytes instead of a heap
 * ByteArray, and stays in canonical GGUF row-major block order — which is
 * what lets a model's quantized payloads never touch the managed heap.
 *
 * GetDirectBufferAddress is a JNI call, so it MUST run before the critical
 * pins (no JNI calls are allowed between Get..Critical and Release..Critical).
 * A NULL address (non-direct buffer) leaves the output untouched; the Kotlin
 * caller guarantees directness by construction (DirectBufferStorage /
 * MappedBufferStorage hand out direct buffers only).
 */
#define SKAINET_JNI_MATMUL_RM_DIRECT_BODY(CALL)                                \
    const uint8_t* w =                                                         \
        (const uint8_t*) (*env)->GetDirectBufferAddress(env, weight);          \
    jfloat* in = w ? (*env)->GetPrimitiveArrayCritical(env, input, NULL) : NULL; \
    jfloat* out = in ? (*env)->GetPrimitiveArrayCritical(env, output, NULL) : NULL; \
    if (out) {                                                                 \
        CALL;                                                                  \
    }                                                                          \
    if (out) (*env)->ReleasePrimitiveArrayCritical(env, output, out, 0);       \
    if (in) (*env)->ReleasePrimitiveArrayCritical(env, input, in, JNI_ABORT);

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q4kMatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q4k_matmul_rm(in, inputOffset, w, weightByteOffset,
                              inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q6kMatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q6k_matmul_rm(in, inputOffset, w, weightByteOffset,
                              inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q80MatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q8_0_matmul_rm(in, inputOffset, w, weightByteOffset,
                               inputDim, outputDim, out, outputOffset))
}

/*
 * Heap-array row-major matmuls (#1193): the same _rm kernels over a pinned
 * ByteArray weight in canonical file order. This is what serves a HEAP-staged
 * canonical weight without a prepack — before these existed, a heap tensor
 * whose format was not mapped-servable fell to the decoding reference kernel
 * silently (measured 48,771 ms/step vs 65 on a mixed-quant model).
 */
JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q4kMatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q4k_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                              inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q6kMatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q6k_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                              inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q80MatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q8_0_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                               inputDim, outputDim, out, outputOffset))
}


JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q40MatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q4_0_matmul_rm(in, inputOffset, w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q40MatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q4_0_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q50MatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q5_0_matmul_rm(in, inputOffset, w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q50MatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5_0_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q51MatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q5_1_matmul_rm(in, inputOffset, w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q51MatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5_1_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q5kMatmulRmDirect(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jobject weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_RM_DIRECT_BODY(
        skainet_q5k_matmul_rm(in, inputOffset, w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_q5kMatmulRm(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_q5k_matmul_rm(in, inputOffset, (const uint8_t*) w, weightByteOffset,
               inputDim, outputDim, out, outputOffset))
}

/*
 * bitnet_gemv (SKEEP-003 §5.3, #1041): int8 activations against ternary TQ2_0
 * weights. Its activation is a *byte* array, not floats, so it does not fit
 * SKAINET_JNI_MATMUL_BODY's float-input shape and pins its three arrays here.
 */
JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_bitnetGemvTq20(
    JNIEnv* env, jobject thiz,
    jbyteArray activation, jint activationOffset, jfloat activationScale,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    jbyte* act = (*env)->GetPrimitiveArrayCritical(env, activation, NULL);
    jbyte* w = act ? (*env)->GetPrimitiveArrayCritical(env, weight, NULL) : NULL;
    jfloat* out = w ? (*env)->GetPrimitiveArrayCritical(env, output, NULL) : NULL;
    if (out) {
        skainet_bitnet_gemv_tq2_0((const int8_t*) act, activationOffset, activationScale,
                                  (const uint8_t*) w, weightByteOffset,
                                  inputDim, outputDim, out, outputOffset);
    }
    if (out) (*env)->ReleasePrimitiveArrayCritical(env, output, out, 0);
    if (w) (*env)->ReleasePrimitiveArrayCritical(env, weight, w, JNI_ABORT);
    if (act) (*env)->ReleasePrimitiveArrayCritical(env, activation, act, JNI_ABORT);
}

/*
 * ternary_f32_gemv (#1139): exact FP32 activations against the sequential
 * BITNET_B1_58 payload — the vendored NeoGPU LUT kernel behind
 * skainet_ternary_f32_gemv. Float input × byte weight × float output, so the
 * shared body fits. The kernel threads internally (pthreads) once
 * outputDim >= 512; the critical-section pins are held for the call's
 * duration either way, same as every other matmul here.
 */
JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_ternaryF32Gemv(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint weightByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    SKAINET_JNI_MATMUL_BODY(
        skainet_ternary_f32_gemv(in, inputOffset, (const uint8_t*) w, weightByteOffset,
                                 inputDim, outputDim, out, outputOffset))
}

/*
 * ternary_lmhead_stage1 (#1150): fused 4-plane BITNET_PLANES lm_head — the
 * vendored NeoGPU kernel behind skainet_ternary_lmhead_stage1. The FP16 row
 * scales live inside the weight buffer, so the shim derives the uint16_t*
 * from the pinned weight array at rowScaleByteOffset (2-byte aligned by the
 * Kotlin seam's contract). Pins three arrays like the bitnet shim above.
 */
JNIEXPORT void JNICALL
Java_sk_ainet_exec_kernel_jni_JniKernels_ternaryLmheadStage1(
    JNIEnv* env, jobject thiz,
    jfloatArray input, jint inputOffset,
    jbyteArray weight, jint planesByteOffset, jint planeStrideBytes, jint rowScaleByteOffset,
    jint inputDim, jint outputDim,
    jfloatArray output, jint outputOffset
) {
    (void) thiz;
    jfloat* in = (*env)->GetPrimitiveArrayCritical(env, input, NULL);
    jbyte* w = in ? (*env)->GetPrimitiveArrayCritical(env, weight, NULL) : NULL;
    jfloat* out = w ? (*env)->GetPrimitiveArrayCritical(env, output, NULL) : NULL;
    if (out) {
        skainet_ternary_lmhead_stage1(
            in, inputOffset,
            (const uint8_t*) w, planesByteOffset, planeStrideBytes,
            (const uint16_t*) ((const uint8_t*) w + rowScaleByteOffset), 0,
            inputDim, outputDim, out, outputOffset);
    }
    if (out) (*env)->ReleasePrimitiveArrayCritical(env, output, out, 0);
    if (w) (*env)->ReleasePrimitiveArrayCritical(env, weight, w, JNI_ABORT);
    if (in) (*env)->ReleasePrimitiveArrayCritical(env, input, in, JNI_ABORT);
}
