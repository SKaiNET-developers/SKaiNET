package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec

/**
 * The f32-activation ternary gemv a platform pack supplies (#1138) — the seam over the vendored
 * NeoGPU LUT kernel (#1137, `skainet_ternary_f32_gemv`).
 *
 * Array-shaped like [BitNetGemvNative] and for the same reason: implementations are FFM, JNI or
 * cinterop shims that pin primitive arrays; [TernaryF32KernelPack] does the view unwrapping once.
 *
 * The weight is the sequential `BITNET_B1_58` **payload** (four codes per byte, low bit-pair
 * first, element order — byte-identical to what [TernaryCodec.encodeBitNet] writes). The
 * per-tensor scale is NOT applied here — the wrapping view kernel owns it. `inputDim` must be a
 * multiple of 4 (the packing is byte-per-4-elements per row).
 */
@ExperimentalMemoryApi
public interface TernaryF32GemvNative {
    /** A name for logs and traces, e.g. `ffm`, `neon`. */
    public val name: String

    /** `out[o] = Σ_k activation(k) · code(o, k)` — one row, scale not applied. */
    public fun gemvPacked(
        activation: FloatArray,
        activationOffset: Int,
        weight: ByteArray,
        weightByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
        out: FloatArray,
        outOffset: Int,
    )
}

/**
 * Installs the exact FP32 × `BITNET_B1_58` path (#1138).
 *
 * Registration happens **only when a native kernel is present**, and that is deliberate: without
 * the LUT kernel the portable way to multiply a ternary weight is the existing int8-requantize →
 * `bitnet_gemv` path ([TernaryKernelPacks]), which stays untouched as the fallback. Registering a
 * Kotlin f32 reference into dispatch would *shadow* that tuned path with a slower exact one; the
 * f32 reference ([TernaryF32GemvKernel]) exists as the correctness oracle and the in-kernel
 * fallback, not as a dispatch entry.
 *
 * Absence of the native artifact is a notice through [warn], never a crash — behavior is exactly
 * today's.
 */
@ExperimentalMemoryApi
public object TernaryF32KernelPack {

    /** What [install] returns when no native kernel is available and nothing was registered. */
    public const val NOT_INSTALLED: String = "ternary_f32_gemv/not-installed"

    /**
     * @param native the platform kernel, or `null` when its artifact is absent
     * @param capabilities what [native] needs; recorded in the key so a device without them never
     *   selects it — the vendored kernel itself needs none beyond baseline NEON
     * @param warn where the "running without the exact f32 path" notice goes
     * @return the name of the kernel that will serve the exact key, or [NOT_INSTALLED]
     */
    public fun install(
        native: TernaryF32GemvNative? = null,
        capabilities: Set<String> = emptySet(),
        warn: (String) -> Unit = {},
    ): String {
        if (native == null) {
            warn(
                "ternary_f32_gemv: no native kernel available — FP32×b1.58 matmuls keep the " +
                    "int8-requantize path. Add the native artifact for the exact f32 path; " +
                    "nothing else changes.",
            )
            return NOT_INSTALLED
        }
        val kernel = NativeTernaryF32ViewKernel(native, TernaryF32GemvKernel.keyFor().copy(capabilities = capabilities))
        KernelDispatch.register(kernel)
        // The dispatcher builds its key from the operands, which say nothing about the CPU — the
        // capability-free key is the reachable one, the capability key documents the requirement.
        // Same two-key pattern as TernaryKernelPacks.install.
        KernelDispatch.register(NativeTernaryF32ViewKernel(native, TernaryF32GemvKernel.keyFor()))
        return kernel.name
    }
}

/**
 * A [ViewKernel] over a [TernaryF32GemvNative]: unwraps the views once, loops the native gemv
 * over the activation rows (prefill included — each row is an independent gemv), and applies the
 * `BITNET_B1_58` per-tensor scale to what the native kernel wrote.
 *
 * Falls back to the reference for anything the native contract does not cover — non-heap storage,
 * a strided view, or `k % 4 != 0` (the sequential packing crosses byte boundaries between rows
 * then) — instead of failing: the fast path is an optimization, never a correctness requirement.
 */
@ExperimentalMemoryApi
public class NativeTernaryF32ViewKernel(
    private val native: TernaryF32GemvNative,
    override val key: KernelKey,
) : ViewKernel {

    override val name: String get() = "ternary_f32_gemv/${native.name}"

    private val reference = TernaryF32GemvKernel(key)

    override fun run(inputs: List<TensorView>, out: TensorView) {
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        val activationFloats = (a.storage as? Storage.Heap)?.floats
        val weightBytes = (w.storage as? Storage.Heap)?.bytes
        val outFloats = (out.storage as? Storage.Heap)?.floats
        if (k % 4 != 0 || activationFloats == null || weightBytes == null || outFloats == null ||
            !a.isContiguous || !out.isContiguous
        ) {
            reference.run(inputs, out)
            return
        }
        if (rows == 0 || n == 0) return
        val aOffset = (a.storage as Storage.Heap).arrayOffset
        val wOffset = (w.storage as Storage.Heap).arrayOffset
        val outOffset = (out.storage as Storage.Heap).arrayOffset
        for (r in 0 until rows) {
            native.gemvPacked(
                activation = activationFloats,
                activationOffset = aOffset + r * k,
                weight = weightBytes,
                weightByteOffset = wOffset,
                inputDim = k,
                outputDim = n,
                out = outFloats,
                outOffset = outOffset + r * n,
            )
        }
        // The native kernel computes the unscaled codes-dot; the per-tensor scale lives in the
        // weight's trailing FP32 and is applied once, here.
        val scale = TernaryCodec.bitNetScale(weightBytes, n * k, wOffset)
        if (scale != 1f) {
            for (i in outOffset until outOffset + rows * n) outFloats[i] *= scale
        }
    }
}
