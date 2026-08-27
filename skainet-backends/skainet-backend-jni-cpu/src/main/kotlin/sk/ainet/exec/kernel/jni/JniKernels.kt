package sk.ainet.exec.kernel.jni

import java.io.File

/**
 * JNI surface over the shared C matmul kernels, plus the two-tier library
 * loader.
 *
 * Two `.so` variants are packaged from the same sources (see
 * `native/CMakeLists.txt`):
 *
 * - `libskainet_jni.so` — baseline `armv8-a`. NEON is architecturally
 *   guaranteed on AArch64, so this runs on every 64-bit ARM core with the
 *   NEON bodies for fp32/q8_0/q4_0/q5k (q4k/q6k fall back to scalar —
 *   their SIMD bodies need the dot-product extension).
 * - `libskainet_jni_v82.so` — `-march=armv8.2-a+fp16+dotprod`, enabling the
 *   `vdotq_s32` paths in q4k/q6k. Executing it on an armv8.0 core
 *   (Cortex-A53, early A55) would SIGILL, so it is only loaded after
 *   `/proc/cpuinfo` confirms `asimddp` (+ `asimdhp`/`fphp`).
 *
 * Exactly ONE variant is loaded per process — both export identical JNI
 * symbols, and selection-at-load is simpler and safer than symbol renaming
 * or ifunc-style per-call dispatch.
 *
 * Method names deliberately contain no underscores: JNI mangles `_` to
 * `_1` in native symbol names, a silent-mismatch trap.
 */
public object JniKernels {

    /**
     * Loaded library variant, or `null` when no variant could be loaded.
     *
     * Initialized EAGERLY in object init, not lazily: Kotlin object
     * initialization runs on first access to ANY member, so a direct call
     * to an `external` function is guaranteed to find the library loaded.
     * (A lazy property would only load when `variant` itself is read —
     * calling `smoke(...)` first would hit `UnsatisfiedLinkError`.)
     */
    public val variant: Variant? = loadVariant()

    public enum class Variant(public val libName: String) {
        /** armv8-a baseline — every AArch64 device, plus x86_64 emulators. */
        BASELINE("skainet_jni"),

        /** armv8.2+dotprod — vdotq_s32 in q4k/q6k; gated on cpuinfo. */
        V82_DOTPROD("skainet_jni_v82"),
    }

    /** Whether a kernel library is loaded and callable. */
    public val isLoaded: Boolean get() = variant != null

    private fun loadVariant(): Variant? {
        if (cpuSupportsV82()) {
            try {
                System.loadLibrary(Variant.V82_DOTPROD.libName)
                return Variant.V82_DOTPROD
            } catch (_: Throwable) {
                // Fall through to baseline — e.g. a packaging that stripped
                // the v82 lib. Never fatal.
            }
        }
        return try {
            System.loadLibrary(Variant.BASELINE.libName)
            Variant.BASELINE
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * `asimddp` is the AArch64 dot-product hwcap; `asimdhp`/`fphp` cover the
     * `+fp16` half of the `-march` the v82 lib was compiled with. Reading
     * `/proc/cpuinfo` is the classic NDK-sanctioned detection path and needs
     * no JNI (which matters: detection must happen BEFORE choosing which
     * library to load). Any read failure means "assume baseline" — never
     * SIGILL on a weird device, just slower q4k/q6k.
     */
    private fun cpuSupportsV82(): Boolean = runCatching {
        val features = File("/proc/cpuinfo").useLines { lines ->
            lines.firstOrNull { it.startsWith("Features") }
        } ?: return false
        "asimddp" in features && ("asimdhp" in features || "fphp" in features)
    }.getOrDefault(false)

    // --- JNI entry points (skainet_jni.c) ---

    /** `skainet_smoke_double`: output[i] = 2 * input[i]. Used by the availability probe. */
    public external fun smoke(input: FloatArray, output: FloatArray, length: Int)

    public external fun q80Matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /**
     * `bitnet_gemv` (#1041): int8 activations with one absmax scale against canonical TQ2_0
     * ternary weights. The activation is a byte array of codes, not floats — a ternary matmul's
     * whole point is that neither operand is wide.
     */
    public external fun bitnetGemvTq20(
        activation: ByteArray, activationOffset: Int, activationScale: Float,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /**
     * `ternary_f32_gemv` (#1139): exact FP32 activations against the sequential
     * BITNET_B1_58 payload — the vendored NeoGPU LUT kernel (#1137). No scale
     * is applied; the caller owns the per-tensor scale. Needs only baseline
     * NEON, so the BASELINE library carries the full SIMD path.
     */
    public external fun ternaryF32Gemv(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /**
     * `ternary_lmhead_stage1` (#1150): fused 4-plane BITNET_PLANES lm_head — the vendored NeoGPU
     * kernel. The FP16 row scales live inside [weight] at [rowScaleByteOffset] (2-byte aligned).
     */
    public external fun ternaryLmheadStage1(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, planesByteOffset: Int, planeStrideBytes: Int, rowScaleByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q40Matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q50Matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q51Matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q4kMatmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q5kMatmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    public external fun q6kMatmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /**
     * Row-major Q4_K matmul over a **direct** [java.nio.ByteBuffer] weight (#1189): the blocks
     * stay in canonical GGUF file order — `(o * blocksPerRow + b) * 144` from [weightByteOffset]
     * — and are read straight out of the buffer's off-heap (typically mmap'd) memory. The buffer
     * MUST be direct; a heap buffer has no native address and the call writes nothing.
     */
    public external fun q4kMatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major Q6_K matmul over a direct ByteBuffer weight (#1189); see [q4kMatmulRmDirect]. */
    public external fun q6kMatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major Q8_0 matmul over a direct ByteBuffer weight (#1192); see [q4kMatmulRmDirect]. */
    public external fun q80MatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /**
     * Row-major matmuls over a **heap** `ByteArray` weight in canonical GGUF file order (#1193):
     * the same `_rm` kernels, pinned-array entry. These serve heap-staged canonical weights
     * without a prepack — the case that used to fall silently to the decoding reference.
     */
    public external fun q4kMatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q6kMatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q80MatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major q40 matmul over a direct ByteBuffer weight (#1192); see [q4kMatmulRmDirect]. */
    public external fun q40MatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q40MatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major q50 matmul over a direct ByteBuffer weight (#1192); see [q4kMatmulRmDirect]. */
    public external fun q50MatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q50MatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major q51 matmul over a direct ByteBuffer weight (#1192); see [q4kMatmulRmDirect]. */
    public external fun q51MatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q51MatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** Row-major q5k matmul over a direct ByteBuffer weight (#1192); see [q4kMatmulRmDirect]. */
    public external fun q5kMatmulRmDirect(
        input: FloatArray, inputOffset: Int,
        weight: java.nio.ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )

    /** See [q4kMatmulRm]. */
    public external fun q5kMatmulRm(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )
}
