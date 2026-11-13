package sk.ainet.backend.api.kernel

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
public object KernelPacks {

    /** Capability marker for a provider that needs an explicit vector unit (Panama, NEON, …). */
    public const val CAPABILITY_VECTOR: String = "vector"

    /**
     * Install the kernels of the best available provider into [KernelDispatch], the packed kernels
     * of every available provider (the best one that carries a format serves it), and the
     * always-present reference kernel for dense FP32. Idempotent per provider name.
     */
    public fun install() {
        install(KernelRegistry.bestAvailable())
        installMissingPacked()
    }

    /**
     * For each packed format that has no feed-order kernel in [KernelDispatch] yet, register the one
     * of the best available provider that carries it. Asking only the best provider left a format
     * it lacks without a kernel although a lower provider has one (#1507). Never replaces a
     * registered kernel, so it is safe to call after providers were added.
     */
    public fun installMissingPacked() {
        for ((encoding, accessor) in packedAccessors) {
            if (KernelDispatch.find(PackedViewMatmulKernel.keyFor(encoding)) != null) continue
            for (p in KernelRegistry.providers()) {
                if (!p.isAvailable()) continue
                val entry = accessor(p) ?: continue
                KernelDispatch.register(viewKernel(p, encoding, entry))
                break
            }
        }
    }

    /** Install the kernels of [provider] only, plus the reference kernel for dense FP32. */
    public fun install(provider: KernelProvider?) {
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
        for ((encoding, accessor) in packedAccessors) {
            val entry = accessor(provider) ?: continue
            KernelDispatch.register(viewKernel(provider, encoding, entry))
        }
    }

    /** A provider's feed-order kernel for one format: the plain entry and, where the SPI has one, the scheduled entry. */
    private class PackedEntry(val plain: PackedMatmulFn, val scheduled: ScheduledPackedMatmulFn? = null)

    private fun viewKernel(provider: KernelProvider, encoding: TensorEncoding, entry: PackedEntry): PackedViewMatmulKernel =
        PackedViewMatmulKernel(provider.name, encoding.name, PackedViewMatmulKernel.keyFor(encoding), entry.plain, entry.scheduled)

    /** Each packed format with the way to ask a provider for its feed-order kernel. */
    private val packedAccessors: List<Pair<TensorEncoding, (KernelProvider) -> PackedEntry?>> = listOf(
        TensorEncoding.Q4_0 to { p -> p.matmulQ4_0()?.let { k -> PackedEntry({ i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) }) } },
        TensorEncoding.Q5_0 to { p -> p.matmulQ5_0()?.let { k -> PackedEntry({ i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) }) } },
        TensorEncoding.Q5_1 to { p -> p.matmulQ5_1()?.let { k -> PackedEntry({ i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) }) } },
        TensorEncoding.Q8_0 to { p -> p.matmulQ8_0()?.let { k -> PackedEntry({ i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) }) } },
        TensorEncoding.Q4_K to { p ->
            p.matmulQ4K()?.let { k ->
                PackedEntry(
                    { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) },
                    { i, io, w, wo, id, od, o, oo, s -> k.matmul(i, io, w, wo, id, od, o, oo, s) },
                )
            }
        },
        TensorEncoding.Q5_K to { p ->
            p.matmulQ5K()?.let { k ->
                PackedEntry(
                    { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) },
                    { i, io, w, wo, id, od, o, oo, s -> k.matmul(i, io, w, wo, id, od, o, oo, s) },
                )
            }
        },
        TensorEncoding.Q6_K to { p -> p.matmulQ6K()?.let { k -> PackedEntry({ i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) }) } },
    )

    /** The reference matmul for dense FP32 — always available, so a key is never unserved. */
    public fun installReference() {
        val dense = OperandKey.contiguous(Format.dense(FP32))
        KernelDispatch.register(ReferenceMatmulKernel(KernelKey("matmul", listOf(dense, dense))))
    }
}

/** The signature the packed SPI matmul kernels share. */
private typealias PackedMatmulFn = (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit

/** The same with the schedule the kernel splits its work under. */
private typealias ScheduledPackedMatmulFn = (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int, sk.ainet.context.schedule.Schedule) -> Unit

/**
 * A [ViewKernel] over an SPI [Fp32MatmulKernel]: both operands dense FP32 and contiguous, weight
 * output-major (`[n, k]`, the shape SKaiNET's dispatch normalises to). Unwraps each view once —
 * per the Phase-2 spike (#1016) — and calls the pack's strided GEMM.
 */
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
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out)
        val oBuf = oHeap.floats ?: return fallback(inputs, out)
        // The SPI GEMM reads the weight input-major: b[p][j] = bBuf[bOffset + p * bStride + j].
        // The dispatcher hands us the weight output-major ([n, k]) — which, when it is a transposed
        // *view* of a contiguous [k, n] buffer, means strides[0] == 1 and strides[1] is that
        // buffer's row stride. Anything else (a genuinely output-major buffer) would need a gather,
        // so it goes to the reference kernel instead of being silently mis-indexed.
        if (b.layout.strides[0] != 1) return fallback(inputs, out)
        val (aBuf, aOffset, aStride) = denseFp32(a) ?: return fallback(inputs, out)
        val (bBuf, bOffset, bStride) = denseFp32(b, requiredStride = b.layout.strides[1]) ?: return fallback(inputs, out)
        kernel.matmul(
            a = aBuf, aOffset = aOffset, aStride = aStride,
            b = bBuf, bOffset = bOffset, bStride = bStride,
            out = oBuf, outOffset = oHeap.arrayOffset + out.layout.offsetElements.toInt(), outStride = out.layout.strides[0],
            m = m, n = n, k = k,
        )
    }

    /**
     * `(FloatArray, offset, stride)` for [view], zero-copy when it is already [Storage.Heap]
     * (the original, only path); otherwise **one bulk snapshot** (never per-element — that is the
     * ~1000×-slower decoding reference path this pack exists to avoid) via [Storage.copyInto],
     * which every storage kind implements uniformly (`SegmentStorage`, `MappedFileStorage`, the
     * NIO-buffer kinds, …). This is the same cost class as [sk.ainet.exec.kernel.FfmRowMajorMatmulKernel]'s
     * existing heap-`ByteArray` staging path — one copy per kernel call, not per element.
     *
     * A weight this small a matmul reaches for is realistically the *whole* backing storage (a
     * top-level loaded tensor, not a narrowed slice) — [requiredStride] guards that assumption:
     * when the view's own row stride does not match what a plain row-major snapshot would produce,
     * this bridge cannot represent it correctly, so it declines (→ fallback) rather than risk a
     * silent-wrong-numbers bug.
     */
    private fun denseFp32(view: TensorView, requiredStride: Int? = null): Triple<FloatArray, Int, Int>? {
        val heap = view.storage as? Storage.Heap
        if (heap != null) {
            val floats = heap.floats ?: return null
            val stride = requiredStride ?: view.layout.strides.getOrElse(0) { 1 }
            return Triple(floats, heap.arrayOffset + view.layout.offsetElements.toInt(), stride)
        }
        if (view.format.dtype != FP32 || view.format.encoding !is sk.ainet.lang.tensor.storage.TensorEncoding.Dense) return null
        val rowStride = requiredStride ?: view.layout.strides.getOrElse(1) { view.shape[view.shape.rank - 1] }
        if (requiredStride == null && !view.isContiguous) return null
        val elementCount = view.elementCount
        val byteWidth = 4
        val byteOffset = view.layout.offsetElements * byteWidth
        val byteLength = (elementCount * byteWidth).toInt()
        if (byteOffset + byteLength > view.storage.sizeBytes) return null
        val bytes = ByteArray(byteLength)
        view.storage.copyInto(bytes, 0, byteOffset, byteLength)
        val floats = FloatArray(elementCount.toInt())
        for (i in floats.indices) {
            val o = i * byteWidth
            val bits = (bytes[o].toInt() and 0xFF) or
                ((bytes[o + 1].toInt() and 0xFF) shl 8) or
                ((bytes[o + 2].toInt() and 0xFF) shl 16) or
                ((bytes[o + 3].toInt() and 0xFF) shl 24)
            floats[i] = Float.fromBits(bits)
        }
        return Triple(floats, 0, rowStride)
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView) {
        ReferenceMatmulKernel(key).run(inputs, out)
    }
}
