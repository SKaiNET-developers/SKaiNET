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
