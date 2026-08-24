package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Wiring between the [KernelProvider] SPI (the platform packs: scalar, Panama/Vector API, native
 * FFM, JNI/NEON) and the view-keyed [KernelDispatch] (SKEEP-003 §5.2).
 *
 * A pack keeps exposing its kernels through `KernelProvider`; this installs them as [ViewKernel]s
 * under the keys the dispatcher looks up, so the generic path gets the fast kernel instead of the
 * decoding reference whenever the operands' formats and layouts match what the pack declares.
 *
 * **The packed kernels are bridged too, now that the order is in the key (#973.2, #1095).** They
 * read their weight input-block-major; a packed `TensorView` from a file is canonical row-major;
 * [sk.ainet.lang.memory.BlockOrder] says which is which, so the kernel declares what it takes and
 * the dispatcher relayouts. That distinction is what #1029 was missing and what made mixing the two
 * a silent-wrong-numbers bug rather than a crash (#968, #971).
 */
@ExperimentalMemoryApi
public object KernelPacks {

    /** Capability marker for a provider that needs an explicit vector unit (Panama, NEON, …). */
    public const val CAPABILITY_VECTOR: String = "vector"

    /**
     * Install the kernels of [provider] (default: the best available one) into [KernelDispatch],
     * plus the always-present reference kernel for dense FP32. Idempotent per provider name.
     */
    public fun install(provider: KernelProvider? = KernelRegistry.bestAvailable()) {
        installReference()
        val p = provider ?: return
        if (!p.isAvailable()) return
        val fp32 = p.matmulFp32() ?: return
        val dense = OperandKey.contiguous(Format.dense(FP32))
        // Two keys, one kernel: a weight normally reaches the dispatcher as a *transposed view* of a
        // contiguous [k, n] buffer — strided by LayoutClass, but exactly what the SPI GEMM's stride
        // arguments express. A contiguous weight is the [n, k]-in-memory case.
        val strided = OperandKey(Format.dense(FP32), LayoutClass.STRIDED)
        KernelDispatch.register(Fp32ViewMatmulKernel(p.name, fp32, KernelKey("matmul", listOf(dense, strided))))
        KernelDispatch.register(Fp32ViewMatmulKernel(p.name, fp32, KernelKey("matmul", listOf(dense, dense))))
        installPacked(p)
    }

    /**
     * Install [provider]'s packed matmul kernels as [PackedViewMatmulKernel]s, keyed on
     * `BLOCKED_INPUT_MAJOR` — the order they actually read.
     *
     * A provider that offers no kernel for a format simply does not get one registered, and the
     * decoding reference kernel keeps serving that format, as it does today.
     */
    public fun installPacked(provider: KernelProvider) {
        fun register(encoding: TensorEncoding, kernel: ((FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit)?) {
            if (kernel == null) return
            KernelDispatch.register(
                PackedViewMatmulKernel(provider.name, encoding.name, PackedViewMatmulKernel.keyFor(encoding), kernel),
            )
        }
        register(TensorEncoding.Q4_0, provider.matmulQ4_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q5_0, provider.matmulQ5_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q5_1, provider.matmulQ5_1()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q8_0, provider.matmulQ8_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q4_K, provider.matmulQ4K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q5_K, provider.matmulQ5K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
        register(TensorEncoding.Q6_K, provider.matmulQ6K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } })
    }

    /** The reference matmul for dense FP32 — always available, so a key is never unserved. */
    public fun installReference() {
        val dense = OperandKey.contiguous(Format.dense(FP32))
        KernelDispatch.register(ReferenceMatmulKernel(KernelKey("matmul", listOf(dense, dense))))
    }
}

/**
 * A [ViewKernel] over an SPI [Fp32MatmulKernel]: both operands dense FP32 and contiguous, weight
 * output-major (`[n, k]`, the shape SKaiNET's dispatch normalises to). Unwraps each view once —
 * per the Phase-2 spike (#1016) — and calls the pack's strided GEMM.
 */
@ExperimentalMemoryApi
public class Fp32ViewMatmulKernel(
    providerName: String,
    private val kernel: Fp32MatmulKernel,
    override val key: KernelKey,
) : ViewKernel {
    override val name: String = "$providerName-fp32"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "matmul takes two operands" }
        val a = inputs[0]; val b = inputs[1]
        val m = a.shape[0]; val k = a.shape[1]; val n = b.shape[0]
        require(b.shape[1] == k) { "inner dimensions disagree: [${m}, ${k}] × [${n}, ${b.shape[1]}]" }
        val aHeap = a.storage as? Storage.Heap ?: return fallback(inputs, out)
        val bHeap = b.storage as? Storage.Heap ?: return fallback(inputs, out)
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out)
        val aBuf = aHeap.floats ?: return fallback(inputs, out)
        val bBuf = bHeap.floats ?: return fallback(inputs, out)
        val oBuf = oHeap.floats ?: return fallback(inputs, out)
        // The SPI GEMM reads the weight input-major: b[p][j] = bBuf[bOffset + p * bStride + j].
        // The dispatcher hands us the weight output-major ([n, k]) — which, when it is a transposed
        // *view* of a contiguous [k, n] buffer, means strides[0] == 1 and strides[1] is that
        // buffer's row stride. Anything else (a genuinely output-major buffer) would need a gather,
        // so it goes to the reference kernel instead of being silently mis-indexed.
        if (b.layout.strides[0] != 1) return fallback(inputs, out)
        kernel.matmul(
            a = aBuf, aOffset = aHeap.arrayOffset + a.layout.offsetElements.toInt(), aStride = a.layout.strides[0],
            b = bBuf, bOffset = bHeap.arrayOffset + b.layout.offsetElements.toInt(), bStride = b.layout.strides[1],
            out = oBuf, outOffset = oHeap.arrayOffset + out.layout.offsetElements.toInt(), outStride = out.layout.strides[0],
            m = m, n = n, k = k,
        )
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView) {
        ReferenceMatmulKernel(key).run(inputs, out)
    }
}
