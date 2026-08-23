package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.types.FP32

/**
 * Wiring between the [KernelProvider] SPI (the platform packs: scalar, Panama/Vector API, native
 * FFM, JNI/NEON) and the view-keyed [KernelDispatch] (SKEEP-003 §5.2).
 *
 * A pack keeps exposing its kernels through `KernelProvider`; this installs them as [ViewKernel]s
 * under the keys the dispatcher looks up, so the generic path gets the fast kernel instead of the
 * decoding reference whenever the operands' formats and layouts match what the pack declares.
 *
 * **Scope note (SKEEP-003 migration, #1029).** Only the dense FP32 kernel is bridged here. The
 * packed (Q4_0…Q6_K) SPI kernels take their weight bytes in **block-major** order — the layout
 * `DefaultCpuOpsBase.transposePackedBlocks` produces — while a packed `TensorView` describes the
 * canonical row-major block order of the file. That contract is exactly what #973 reports as
 * unwritten and contradictory across the engine and the converters, and getting it wrong is the
 * silent-wrong-numbers class of #968/#971. Bridging the packed kernels therefore waits until #973
 * pins the byte order down; until then the packed fast paths stay on their existing (working)
 * ladder in `DefaultCpuOps`/`DefaultCpuOpsJvm`, and the registry serves packed operands with the
 * decoding reference kernel, which is correct for any layout.
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
