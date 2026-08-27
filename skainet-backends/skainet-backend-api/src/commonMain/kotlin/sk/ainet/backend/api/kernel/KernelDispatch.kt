package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.memory.trace.kernel as traceKernel
import sk.ainet.lang.tensor.Shape

/**
 * Kernel selection on declared descriptors instead of an `is`-ladder over Kotlin classes
 * (SKEEP-003 §5.1). The order is: **normalize** the operands as zero-copy views (so rank-1 decode
 * steps never reach a kernel written for rank 2 — the #993 root cause disappears), build the
 * [KernelKey], look it up, insert **visible** adapters when a kernel cannot take an operand as it
 * is, and fall back to the reference kernel, which is correct for every format because it decodes.
 *
 * Adapters allocate in the caller's [Scope] (a `Forward` scope in a generation loop) and are
 * emitted as [TraceEvent.AdapterInserted] — the "hidden 12 GB" of #782 becomes a visible event.
 */
@ExperimentalMemoryApi
public object KernelDispatch {

    private val kernels: MutableList<ViewKernel> = mutableListOf()

    /** Register [kernel]; later registrations win for the same key (a pack can override the reference). */
    public fun register(kernel: ViewKernel) {
        kernels.removeAll { it.key == kernel.key && it.name == kernel.name }
        kernels.add(0, kernel)
    }

    /** Every registered kernel, most recently registered first. */
    public fun kernels(): List<ViewKernel> = kernels.toList()

    /** The kernel registered for [key], or `null`. */
    public fun find(key: KernelKey): ViewKernel? = kernels.firstOrNull { it.key == key }

    public fun clearForTesting() { kernels.clear() }

    /**
     * Normalize a matmul operand pair to rank 2 as **views** (rule 5, §5.1 "rank handling happens
     * once"): `[k]` becomes `[1, k]`, `[b, s, k]` becomes `[b*s, k]` when contiguous. Returns the
     * normalized activation and the number of leading dims that were flattened, so the caller can
     * reshape the result back.
     */
    public fun normalizeActivation(a: TensorView): Pair<TensorView, IntArray> = when {
        a.shape.rank == 1 -> a.unsqueeze(0) to intArrayOf()
        a.shape.rank == 2 -> a to intArrayOf()
        else -> {
            val leading = IntArray(a.shape.rank - 1) { a.shape[it] }
            require(a.isContiguous) { "flattening leading dims needs a contiguous activation; materialize first" }
            var rows = 1
            for (d in leading) rows *= d
            a.reshapeContiguous(Shape(rows, a.shape[a.shape.rank - 1])) to leading
        }
    }

    /**
     * Select and run `matmul(a, b)`, writing into [out]. [scope] owns any adapter the selection
     * needs; [sink] sees the kernel run and every adapter.
     *
     * @throws UnsupportedKernelException when neither a kernel nor the reference path can serve the key
     */
    public fun matmul(
        a: TensorView,
        b: TensorView,
        out: TensorView,
        scope: Scope = Scope.Ambient,
        sink: TraceSink = NoopTraceSink,
        /**
         * Relayout a canonical packed weight into kernel order when that is what unlocks a packed
         * kernel (#973/#1095).
         *
         * **Off by default, deliberately.** The relayout is O(bytes), so doing it inside a decode
         * step would copy the whole weight on every token — the per-forward copy #973 objects to,
         * merely moved. A weight is prepacked *once*, at load
         * ([sk.ainet.lang.memory.TensorView.prepack]); a canonical weight handed straight to the
         * dispatcher gets the decoding reference kernel, which is correct and slower. Pass `true`
         * for a one-shot call where the copy is cheaper than the reference path.
         */
        prepackWeights: Boolean = false,
    ) {
        val key = KernelKey.matmul(a, b)
        val exact = find(key)
        if (exact != null) {
            runTraced(exact, listOf(a, b), out, sink)
            return
        }
        // The weight's encoding may *ask* for a different activation format — a ternary weight wants
        // int8 with a per-token scale (`W1.58A8`, §5.3). Honour the request when a kernel exists for
        // the requantized pair: the adapter costs bytes in the caller's scope every step, so it is
        // allocated there and emitted as an AdapterInserted rather than hidden inside the kernel.
        val wanted = b.format.encoding.blockSpec?.activation
        if (wanted != null && wanted != a.format) {
            val requantized = requantizeFor(wanted, a, scope, sink)
            if (requantized != null) {
                val ternaryKernel = find(KernelKey.matmul(requantized, b))
                if (ternaryKernel != null) {
                    runTraced(ternaryKernel, listOf(requantized, b), out, sink)
                    return
                }
            }
        }
        // A packed kernel reads its weight input-block-major; a weight loaded from a file is
        // canonical. Now that the order is in the key (#973/#1094) the two can be bridged — but the
        // relayout is O(bytes), so it happens only when the caller asks, and the right shape for a
        // hot loop is a weight prepacked once at load, which hits the exact key above and copies
        // nothing.
        if (prepackWeights && b.layout.blocked && b.layout.blockOrder == BlockOrder.ROW_MAJOR) {
            val prepacked = b.prepack(BlockOrder.INPUT_BLOCK_MAJOR, scope, sink)
            val packedKernel = find(KernelKey.matmul(a, prepacked))
            if (packedKernel != null) {
                runTraced(packedKernel, listOf(a, prepacked), out, sink)
                return
            }
        }
        // No exact kernel: adapt the operands a kernel would accept, then fall back to the reference,
        // which reads any format through decoding get().
        val adaptedA = adapt(a, scope, sink, "gather")
        val reference = ReferenceMatmulKernel(KernelKey.matmul(adaptedA, b))
        runTraced(reference, listOf(adaptedA, b), out, sink)
    }

    /**
     * Convert [activation] into the [wanted] activation format, or `null` when no adapter for it
     * exists. Today the only one is the int8 absmax requantization the ternary kernels ask for.
     */
    private fun requantizeFor(wanted: Format, activation: TensorView, scope: Scope, sink: TraceSink): TensorView? =
        if (wanted == sk.ainet.lang.memory.I8Absmax.FORMAT && activation.shape.rank == 2) {
            sk.ainet.lang.memory.I8Absmax.requantize(activation, scope, sink)
        } else {
            null
        }

    /** Materialize [view] into a dense contiguous view when it is strided; emits an adapter event. */
    public fun adapt(view: TensorView, scope: Scope, sink: TraceSink, kind: String): TensorView {
        if (view.isContiguous || view.layout.blocked) return view
        val dense = view.materialize(Format.dense(view.format.dtype), scope)
        if (sink.isEnabled) {
            sink.emit(TraceEvent.AdapterInserted(kind, view.format, dense.format, dense.elementCount * view.format.dtype.sizeInBytes, view.id))
        }
        return dense
    }

    private fun runTraced(kernel: ViewKernel, inputs: List<TensorView>, out: TensorView, sink: TraceSink) {
        if (!sink.isEnabled) { kernel.run(inputs, out); return }
        sink.traceKernel(
            op = kernel.key.op,
            kernel = kernel.name,
            inputs = inputs.map { it.id },
            output = out.id,
            bytesRead = inputs.sumOf { it.elementCount * it.format.dtype.sizeInBytes },
            bytesWritten = out.elementCount * out.format.dtype.sizeInBytes,
        ) { kernel.run(inputs, out, sink) }
    }
}

/** A view of the same contiguous bytes under a different shape (rule 5: reshape is a view). */
@ExperimentalMemoryApi
public fun TensorView.reshapeContiguous(newShape: Shape): TensorView {
    require(isContiguous) { "reshape needs a contiguous view" }
    require(newShape.volume.toLong() == elementCount) { "reshape must keep the element count ($elementCount), got ${newShape.volume}" }
    return TensorView(newShape, format, sk.ainet.lang.memory.Layout.rowMajor(newShape, format, layout.offsetElements), storage, id)
}
