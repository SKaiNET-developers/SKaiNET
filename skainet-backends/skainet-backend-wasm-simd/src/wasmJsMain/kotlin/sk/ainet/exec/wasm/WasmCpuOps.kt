package sk.ainet.exec.wasm

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.tensor.ops.DefaultCpuOpsBase
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.FusedNormOps
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * The browser tier's `TensorOps` (SKEEP-007 §T1 and the resident-activation step): [DefaultCpuOpsBase]
 * with the dense matmuls, the elementwise arithmetic, the activations and the reductions routed
 * through the SIMD128 sidecar, and their outputs kept **resident** in the sidecar's linear memory
 * ([WasmLinearTensorData]) so a chain of them never copies an activation back to the Kotlin heap
 * between kernels. Packed weights need no override — they reach the pack through `KernelDispatch`
 * like on every other tier. Everything else is the common scalar implementation, which reads a
 * resident tensor through its heap mirror (one bulk download, the first time), and so is
 * everything here while the sidecar is not ready.
 *
 * What goes to the sidecar, and when:
 *
 * - `matmul`, `matmulWeightTransposed` (dense FP32): always; the output is resident.
 * - `add`/`subtract`/`multiply`/`divide`: when at least one operand is already resident. Two heap
 *   operands stay on the common path — uploading both costs more than the scalar loop saves, and
 *   a chain starts resident at its first matmul anyway.
 * - `sigmoid`/`silu`/`gelu`/`erf`/`tanh`: always — the transcendental loop is several times the
 *   upload. `relu`, `sum`, `mean`: when the operand is resident.
 * - `softmax`/`logSoftmax` over the last dimension, `scaledDotProductAttention` and the fused
 *   normalisations of [FusedNormOps] (what `LayerNormalization` calls): always — one sidecar pass
 *   against several allocating scalar passes, whatever the operands' home.
 * - `conv1d`/`conv2d`/`conv3d` (dense FP32): always — im2col and the GEMM in the sidecar, the
 *   weight resident after its first call, the output resident.
 *
 * An output is resident only while [WasmSimd.residentBudgetBytes] allows; above it the kernel still
 * runs in the sidecar and the result is read back to the heap. Scoped contexts (`ForwardScope`)
 * allocate the resident output from the scope, so `reset()` frees the sidecar bytes; ambient outputs
 * are freed when their tensor is collected (see [WasmLinearStorage]).
 */
public class WasmCpuOps(
    dataFactory: TensorDataFactory,
    schedule: Schedule,
) : DefaultCpuOpsBase(dataFactory, schedule), FusedNormOps {

    override fun withSchedule(schedule: Schedule): TensorOps =
        if (schedule === this.schedule) this else WasmCpuOps(dataFactory, schedule)

    /** The sidecar computes `x · Wᵀ` from the weight's rows itself; no Kotlin-side row-dot and no transposed copy. */
    override fun preferDirectWeightTransposed(): Boolean = false

    // ── operands and outputs ──────────────────────────────────────────────────────────────────────

    /** A sidecar address for an operand: a resident tensor's own, or a staged copy freed after the call. */
    private class Operand(val ptr: Int, val temp: Boolean)

    /** Dense FP32 in sidecar memory: the storage and the byte offset of element 0 in it. */
    private class Sidecar(val storage: WasmLinearStorage, val byteOffset: Int) {
        val isResident: Boolean get() = storage.isAlive
        fun pointer(): Int = storage.pointer() + byteOffset
    }

    /**
     * Where [t]'s floats are in the sidecar, when its view is dense FP32, gap-free and over a
     * [WasmLinearStorage]; else `null`. Read off the view, not the class (SKEEP-003, #1507), so
     * a contiguous slice of a resident tensor is resident too.
     *
     * Every caller asks this before `heapFloats`: on resident data that is the heap mirror, and
     * asking for it downloads the tensor.
     */
    private fun sidecar(t: Tensor<*, *>): Sidecar? {
        val view = t.data.view ?: return null
        val storage = view.storage as? WasmLinearStorage ?: return null
        if (view.format != DENSE_FP32 || !view.isContiguous) return null
        return Sidecar(storage, (view.layout.offsetElements * 4).toInt())
    }

    private fun isResident(t: Tensor<*, *>): Boolean = sidecar(t)?.isResident == true

    /** Dense FP32 the sidecar can take: resident, a heap array, or a slab window. */
    private fun isSidecarFloat(t: Tensor<*, *>): Boolean =
        t.dtype == FP32::class && (sidecar(t) != null || t.data.heapFloats != null)

    private fun operand(t: Tensor<*, *>): Operand? {
        val resident = sidecar(t)
        if (resident != null) return if (resident.isResident) Operand(resident.pointer(), false) else null
        val floats = t.data.heapFloats ?: return null
        return stage(floats, t.data.heapFloatsOffset, t.shape.volume)
    }

    /**
     * The heap mirror behind a dense view over sidecar memory, for the common fast paths that
     * reach a resident tensor through a slice (the slice has no `heapFloats` of its own).
     */
    override fun offHeapFloatWindow(view: sk.ainet.lang.memory.TensorView): FloatWindow? {
        val storage = view.storage as? WasmLinearStorage ?: return null
        if (!storage.isAlive) return null
        return FloatWindow(storage.heapMirror(), view.layout.offsetElements.toInt())
    }

    /**
     * A sidecar address for a weight. A heap weight in an array of its own is a parameter: it is
     * uploaded once and stays (re-uploading it per call made single-token decode run at copy
     * speed). A resident weight is read where it is. A window of a larger array, and anything a
     * forward scope owns, is staged for the call: the upload is kept by the array's identity, and
     * a forward slab holds other values after its next reset.
     */
    private fun weightOperand(t: Tensor<*, *>): Operand? {
        if (sidecar(t) != null) return operand(t)
        val floats = floatsFromZero(t.data)
        if (floats != null && t.data.view?.storage?.scope != ScopeKind.FORWARD) {
            return Operand(WasmKernels.denseWeightHandle(floats), false)
        }
        return operand(t)
    }

    private fun stage(src: FloatArray, offset: Int, count: Int): Operand {
        val ptr = WasmKernels.alloc(maxOf(count, 1) * 4)
        if (count > 0) WasmKernels.writeFloats(ptr, src, offset, count)
        return Operand(ptr, true)
    }

    private fun release(o: Operand?) {
        if (o != null && o.temp) WasmKernels.free(o.ptr)
    }

    /** A resident output allocation from the active scope (ambient: GC-managed), or `null` when the scope cannot place it in the sidecar. */
    private fun residentStorage(bytes: Long): WasmLinearStorage? {
        val scope = dispatchScope()
        return if (scope === Scope.Ambient) WasmLinearStorage.allocate(bytes, ScopeKind.AMBIENT, zero = false)
        else scope.allocate(bytes, MemoryDomain.HOST_OFFHEAP) as? WasmLinearStorage
    }

    /**
     * Run [fill] into an output buffer of [shape]: resident when the budget allows, otherwise a
     * temporary that is read back to the heap and adopted through the factory.
     */
    @Suppress("UNCHECKED_CAST")
    private inline fun <T : DType, V> output(
        shape: Shape,
        dtype: KClass<T>,
        inputs: Array<out Tensor<T, V>>,
        fill: (outPtr: Int) -> Unit,
    ): Tensor<T, V> {
        val n = shape.volume
        val bytes = n.toLong() * 4
        if (WasmSimd.residentBudgetBytes - WasmKernels.residentActivationBytes() >= bytes) {
            residentStorage(bytes)?.let { storage ->
                fill(storage.pointer())
                return newTensor(WasmLinearTensorData<T>(shape, storage) as TensorData<T, V>, dtype, *inputs)
            }
        }
        val tmp = WasmKernels.alloc(maxOf(n, 1) * 4)
        try {
            fill(tmp)
            val out = FloatArray(n)
            if (n > 0) WasmKernels.readFloats(tmp, out, 0, n)
            return floatResult(shape, dtype, out, *inputs)
        } finally {
            WasmKernels.free(tmp)
        }
    }

    /**
     * A resident copy of a dense FP32 [tensor] — the explicit "upload once" for an input that many
     * sidecar ops will read (a bench's operands, a prompt's embeddings), so none of them stages it
     * per call. The tensor itself if it is already resident; the common data unchanged when the
     * sidecar is not ready or the budget is spent.
     */
    public fun <T : DType, V> toResident(tensor: Tensor<T, V>): Tensor<T, V> {
        if (!WasmKernels.isReady || !isSidecarFloat(tensor) || isResident(tensor)) return tensor
        val n = tensor.shape.volume
        val o = operand(tensor) ?: return tensor
        try {
            val src = o.ptr
            val bytes = n.toLong() * 4
            if (WasmSimd.residentBudgetBytes - WasmKernels.residentActivationBytes() < bytes) return tensor
            val storage = residentStorage(bytes) ?: return tensor
            WasmKernels.copy(src, storage.pointer(), n * 4)
            @Suppress("UNCHECKED_CAST")
            return newTensor(WasmLinearTensorData<T>(tensor.shape, storage) as TensorData<T, V>, tensor.dtype, tensor)
        } finally {
            release(o)
        }
    }

    // ── matmuls ───────────────────────────────────────────────────────────────────────────────────

    /**
     * Dense `A(m,k) · B(k,n)` on FP32 operands runs the sidecar GEMM. The common implementation
     * serves this shape with its own `[k, n]` loop before it consults the registry, so without this
     * override the SIMD128 kernel was reachable only through `KernelDispatch`. Resident operands are
     * read in place; heap operands are staged per call (`B` is not necessarily a parameter). A packed
     * weight is not dense FP32 data and keeps its path.
     */
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && a.rank == 2 && b.rank == 2 && isSidecarFloat(a) && isSidecarFloat(b) && untransposedWeight(b) == null) {
            val m = a.shape[0]
            val k = a.shape[1]
            val n = b.shape[1]
            require(k == b.shape[0]) { "Matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}" }
            if (m > 0 && n > 0) {
                var ao: Operand? = null
                var bo: Operand? = null
                try {
                    ao = operand(a) ?: return super.matmul(a, b)
                    bo = operand(b) ?: return super.matmul(a, b)
                    val aPtr = ao.ptr
                    val bPtr = bo.ptr
                    return output(Shape(m, n), a.dtype, arrayOf(a, b)) { out -> WasmKernels.fp32Matmul(aPtr, k, bPtr, n, out, n, m, n, k) }
                } finally {
                    release(bo)
                    release(ao)
                }
            }
        }
        return super.matmul(a, b)
    }

    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        if (WasmKernels.isReady && weight.shape.rank == 2 && x.rank >= 1 && isSidecarFloat(x) && isSidecarFloat(weight)) {
            val n = weight.shape[0]
            val k = weight.shape[1]
            if (x.shape.dimensions.last() == k) {
                val m = if (k == 0) 0 else x.shape.volume / k
                if (m > 0 && n > 0) {
                    var xo: Operand? = null
                    var wo: Operand? = null
                    try {
                        xo = operand(x) ?: return super.matmulWeightTransposed(x, weight)
                        wo = weightOperand(weight) ?: return super.matmulWeightTransposed(x, weight)
                        val xPtr = xo.ptr
                        val wPtr = wo.ptr
                        val outShape = if (x.rank == 1) Shape(n) else Shape(*(x.shape.dimensions.copyOf(x.rank - 1) + n))
                        return output(outShape, x.dtype, arrayOf(x, weight)) { out -> WasmKernels.fp32MatmulWt(xPtr, wPtr, out, m, n, k) }
                    } finally {
                        release(wo)
                        release(xo)
                    }
                }
            }
        }
        return super.matmulWeightTransposed(x, weight)
    }

    // ── elementwise ───────────────────────────────────────────────────────────────────────────────

    /** The operand's index mode against the broadcast output shape, or `null` for a broadcast the kernel does not take. */
    private fun mode(s: Shape, out: Shape): Int? = when {
        s == out -> WasmKernels.F32_FULL
        s.volume == 1 -> WasmKernels.F32_SCALAR
        s.rank >= 1 && out.rank >= 1 && s[s.rank - 1] == out[out.rank - 1] && s.dimensions.dropLast(1).all { it == 1 } -> WasmKernels.F32_BIAS
        else -> null
    }

    private fun <T : DType, V> binary(op: Int, a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!WasmKernels.isReady || a.dtype != b.dtype || !isSidecarFloat(a) || !isSidecarFloat(b)) return null
        if (!isResident(a) && !isResident(b)) return null
        val outShape = try {
            broadcastShapes(a.shape, b.shape)
        } catch (e: IllegalArgumentException) {
            return null
        }
        val n = outShape.volume
        if (n == 0) return null
        val aMode = mode(a.shape, outShape) ?: return null
        val bMode = mode(b.shape, outShape) ?: return null
        val inner = outShape[outShape.rank - 1]
        var ao: Operand? = null
        var bo: Operand? = null
        try {
            ao = operand(a) ?: return null
            bo = operand(b) ?: return null
            val aPtr = ao.ptr
            val bPtr = bo.ptr
            return output(outShape, a.dtype, arrayOf(a, b)) { out -> WasmKernels.f32Binary(op, aPtr, aMode, bPtr, bMode, out, n, inner) }
        } finally {
            release(bo)
            release(ao)
        }
    }

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_ADD, a, b) ?: super.add(a, b)

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_SUB, a, b) ?: super.subtract(a, b)

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_MUL, a, b) ?: super.multiply(a, b)

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> =
        binary(WasmKernels.F32_DIV, a, b) ?: super.divide(a, b)

    // ── activations ───────────────────────────────────────────────────────────────────────────────

    private fun <T : DType, V> unary(op: Int, t: Tensor<T, V>, heapToo: Boolean): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(t)) return null
        if (!heapToo && !isResident(t)) return null
        val n = t.shape.volume
        if (n == 0) return null
        val o = operand(t) ?: return null
        try {
            val inPtr = o.ptr
            return output(t.shape, t.dtype, arrayOf(t)) { out -> WasmKernels.f32Unary(op, inPtr, out, n) }
        } finally {
            release(o)
        }
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_RELU, tensor, heapToo = false) ?: super.relu(tensor)

    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_SIGMOID, tensor, heapToo = true) ?: super.sigmoid(tensor)

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_SILU, tensor, heapToo = true) ?: super.silu(tensor)

    override fun <T : DType, V> gelu(tensor: Tensor<T, V>, approximate: Boolean): Tensor<T, V> =
        unary(if (approximate) WasmKernels.F32_GELU_TANH else WasmKernels.F32_GELU_EXACT, tensor, heapToo = true)
            ?: super.gelu(tensor, approximate)

    override fun <T : DType, V> erf(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_ERF, tensor, heapToo = true) ?: super.erf(tensor)

    override fun <T : DType, V> tanh(tensor: Tensor<T, V>): Tensor<T, V> =
        unary(WasmKernels.F32_TANH, tensor, heapToo = true) ?: super.tanh(tensor)

    // ── reductions ────────────────────────────────────────────────────────────────────────────────

    private fun <T : DType, V> reduce(t: Tensor<T, V>, dim: Int?, mean: Boolean): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(t) || !isResident(t)) return null
        val rank = t.rank
        val volume = t.shape.volume
        if (volume == 0) return null
        val o = operand(t) ?: return null
        try {
            val inPtr = o.ptr
            if (dim == null) {
                return output(Shape(), t.dtype, arrayOf(t)) { out -> WasmKernels.f32ReduceAll(inPtr, volume, mean, out) }
            }
            val nd = if (dim < 0) dim + rank else dim
            if (nd !in 0 until rank) return null
            val dims = t.shape.dimensions
            var outer = 1
            for (i in 0 until nd) outer *= dims[i]
            val red = dims[nd]
            var inner = 1
            for (i in nd + 1 until rank) inner *= dims[i]
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) { if (i == nd) continue; outDims[oi++] = dims[i] }
            return output(Shape(outDims), t.dtype, arrayOf(t)) { out -> WasmKernels.f32ReduceAxis(inPtr, outer, red, inner, mean, out) }
        } finally {
            release(o)
        }
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> =
        reduce(tensor, dim, mean = false) ?: super.sum(tensor, dim)

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> =
        reduce(tensor, dim, mean = true) ?: super.mean(tensor, dim)

    // ── fused row kernels: softmax, the normalisations, attention ─────────────────────────────────

    private fun <T : DType, V> softmaxRows(t: Tensor<T, V>, dim: Int, log: Boolean): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(t)) return null
        val rank = t.rank
        if (rank == 0) return null
        val nd = if (dim < 0) dim + rank else dim
        if (nd != rank - 1) return null
        val cols = t.shape[nd]
        val n = t.shape.volume
        if (cols <= 0 || n == 0) return null
        val o = operand(t) ?: return null
        try {
            val inPtr = o.ptr
            return output(t.shape, t.dtype, arrayOf(t)) { out -> WasmKernels.f32SoftmaxRows(inPtr, out, n / cols, cols, log) }
        } finally {
            release(o)
        }
    }

    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> =
        softmaxRows(tensor, dim, log = false) ?: super.softmax(tensor, dim)

    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> =
        softmaxRows(tensor, dim, log = true) ?: super.logSoftmax(tensor, dim)

    /** A norm parameter of [cols] values as a sidecar operand: in place or staged when dense FP32, copied out otherwise (a placeholder). */
    private fun normParam(t: Tensor<*, *>, cols: Int): Operand? {
        if (t.shape.volume != cols) return null
        if (t.dtype == FP32::class) operand(t)?.let { return it }
        return stage(t.data.copyToFloatArray(), 0, cols)
    }

    private fun <T : DType, V> norm(
        input: Tensor<T, V>, gamma: Tensor<T, V>?, beta: Tensor<T, V>?, eps: Float, normalizedRank: Int, rms: Boolean,
    ): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(input)) return null
        val rank = input.rank
        if (normalizedRank !in 1..rank) return null
        var cols = 1
        for (i in rank - normalizedRank until rank) cols *= input.shape[i]
        val n = input.shape.volume
        if (cols == 0 || n == 0) return null
        val rows = n / cols
        var io: Operand? = null
        var go: Operand? = null
        var bo: Operand? = null
        try {
            io = operand(input) ?: return null
            go = gamma?.let { normParam(it, cols) ?: return null }
            bo = beta?.let { normParam(it, cols) ?: return null }
            val inPtr = io.ptr
            val gPtr = go?.ptr ?: 0
            val bPtr = bo?.ptr ?: 0
            val inputs = listOfNotNull(input, gamma, beta).toTypedArray()
            return output(input.shape, input.dtype, inputs) { out ->
                if (rms) WasmKernels.f32RmsNorm(inPtr, gPtr, out, rows, cols, eps)
                else WasmKernels.f32LayerNorm(inPtr, gPtr, bPtr, out, rows, cols, eps)
            }
        } finally {
            release(bo)
            release(go)
            release(io)
        }
    }

    override fun <T : DType, V> layerNorm(input: Tensor<T, V>, gamma: Tensor<T, V>?, beta: Tensor<T, V>?, eps: Float, normalizedRank: Int): Tensor<T, V>? =
        norm(input, gamma, beta, eps, normalizedRank, rms = false)

    override fun <T : DType, V> rmsNorm(input: Tensor<T, V>, gamma: Tensor<T, V>?, eps: Float, normalizedRank: Int): Tensor<T, V>? =
        norm(input, gamma, null, eps, normalizedRank, rms = true)

    /**
     * Attention in one sidecar pass (`skainet_f32_sdpa`): the same operand contract and mask rules
     * as the common implementation, the scores never materialised on the Kotlin heap, query rows
     * threaded in the sidecar's pool. Resident operands are read in place — a chain
     * `q/k/v = x · Wᵀ → sdpa → · Wₒᵀ` never leaves linear memory.
     */
    override fun <T : DType, V> scaledDotProductAttention(
        query: Tensor<T, V>,
        key: Tensor<T, V>,
        value: Tensor<T, V>,
        mask: Tensor<T, V>?,
        scale: Float,
        causal: Boolean,
        keyLengths: Tensor<DType, *>?,
        window: Int,
        rightContext: Int,
    ): Tensor<T, V> {
        if (!WasmKernels.isReady || !isSidecarFloat(query) || !isSidecarFloat(key) || !isSidecarFloat(value) || (mask != null && !isSidecarFloat(mask))) {
            return super.scaledDotProductAttention(query, key, value, mask, scale, causal, keyLengths, window, rightContext)
        }
        requireSdpaOperands(query, key, value, window, rightContext)
        val batch = query.shape[0]
        val heads = query.shape[1]
        val seqQ = query.shape[2]
        val headDim = query.shape[3]
        val kvHeads = key.shape[1]
        val seqKV = key.shape[2]
        val lengths = sdpaKeyLengths(keyLengths, batch, seqKV)
        val maskDims = sdpaMaskDims(mask, batch, heads, kvHeads, seqQ, seqKV) ?: intArrayOf(1, 1, 1, 1)
        val effectiveScale = if (scale == 0f) (1.0 / kotlin.math.sqrt(headDim.toDouble())).toFloat() else scale
        var qo: Operand? = null
        var ko: Operand? = null
        var vo: Operand? = null
        var mo: Operand? = null
        var lengthsPtr = 0
        try {
            qo = operand(query) ?: return super.scaledDotProductAttention(query, key, value, mask, scale, causal, keyLengths, window, rightContext)
            ko = operand(key) ?: return super.scaledDotProductAttention(query, key, value, mask, scale, causal, keyLengths, window, rightContext)
            vo = operand(value) ?: return super.scaledDotProductAttention(query, key, value, mask, scale, causal, keyLengths, window, rightContext)
            mo = mask?.let { operand(it) ?: return super.scaledDotProductAttention(query, key, value, mask, scale, causal, keyLengths, window, rightContext) }
            if (lengths != null && batch > 0) {
                lengthsPtr = WasmKernels.alloc(batch * 4)
                WasmKernels.writeInts(lengthsPtr, lengths, 0, batch)
            }
            val qPtr = qo.ptr
            val kPtr = ko.ptr
            val vPtr = vo.ptr
            val mPtr = mo?.ptr ?: 0
            val lPtr = lengthsPtr
            return output(Shape(batch, heads, seqQ, headDim), query.dtype, arrayOf(query, key, value)) { out ->
                WasmKernels.f32Sdpa(
                    qPtr, kPtr, vPtr, out, batch, heads, kvHeads, seqQ, seqKV, headDim,
                    effectiveScale, causal, window, rightContext, lPtr, mPtr, maskDims[0], maskDims[1], maskDims[2], maskDims[3],
                )
            }
        } finally {
            if (lengthsPtr != 0) WasmKernels.free(lengthsPtr)
            release(mo)
            release(vo)
            release(ko)
            release(qo)
        }
    }

    // ── convolutions: im2col + GEMM in the sidecar ────────────────────────────────────────────────

    /**
     * `conv1d` / `conv2d` / `conv3d` on dense FP32 operands run `skainet_f32_conv`: the receptive
     * fields are gathered patch-major into a sidecar scratch and multiplied against the weight rows
     * by the SIMD128 `x · Wᵀ` kernel, both threaded over patches — the JVM's im2col + Vector API
     * path, in the sidecar. The weight is a parameter and stays resident after its first call; a heap
     * input is staged per call and a resident one read in place; the output is resident, so a
     * `conv → relu → …` chain never leaves linear memory. Groups (depthwise included), stride,
     * padding and dilation go to the kernel as they are, a bias of `[c_out]` or `[1, c_out, 1, …]`
     * is added there. Whatever the common validation would reject stays on the common path, which
     * raises its message; `null` here means "not this path".
     */
    private fun <T : DType, V> conv(
        input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?,
        spatial: Int, stride: IntArray, padding: IntArray, dilation: IntArray, groups: Int,
    ): Tensor<T, V>? {
        if (!WasmKernels.isReady || !isSidecarFloat(input) || !isSidecarFloat(weight)) return null
        if (bias != null && !isSidecarFloat(bias)) return null
        val rank = spatial + 2
        if (input.rank != rank || weight.rank != rank || groups < 1) return null
        val n = input.shape[0]
        val cIn = input.shape[1]
        val cOut = weight.shape[0]
        if (n <= 0 || cOut <= 0 || cIn % groups != 0 || cOut % groups != 0 || weight.shape[1] != cIn / groups) return null
        if (bias != null) {
            val ok = (bias.rank == 1 && bias.shape[0] == cOut) || (bias.rank == rank && bias.shape[1] == cOut && bias.shape.volume == cOut)
            if (!ok) return null
        }
        // Three spatial dims for the kernel; a lower rank fills the leading ones with the identity geometry.
        val inDims = IntArray(3) { 1 }
        val k = IntArray(3) { 1 }
        val s = IntArray(3) { 1 }
        val p = IntArray(3)
        val d = IntArray(3) { 1 }
        val outDims = IntArray(3) { 1 }
        for (i in 0 until spatial) {
            val j = 3 - spatial + i
            inDims[j] = input.shape[2 + i]
            k[j] = weight.shape[2 + i]
            s[j] = stride[i]
            p[j] = padding[i]
            d[j] = dilation[i]
            if (s[j] <= 0 || d[j] <= 0 || k[j] <= 0 || p[j] < 0) return null
            outDims[j] = (inDims[j] + 2 * p[j] - d[j] * (k[j] - 1) - 1) / s[j] + 1
            if (outDims[j] <= 0) return null
        }
        val patchCount = outDims[0].toLong() * outDims[1] * outDims[2]
        val colSize = (cIn / groups).toLong() * k[0] * k[1] * k[2]
        val colBytes = patchCount * colSize * 4
        if (colSize <= 0 || colBytes > Int.MAX_VALUE) return null
        val outShape = Shape(*(intArrayOf(n, cOut) + outDims.copyOfRange(3 - spatial, 3)))
        val geo = intArrayOf(
            n, cIn, cOut, groups,
            inDims[0], inDims[1], inDims[2], k[0], k[1], k[2], s[0], s[1], s[2],
            p[0], p[1], p[2], d[0], d[1], d[2], outDims[0], outDims[1], outDims[2],
        )
        check(geo.size == WasmKernels.F32_CONV_GEO_INTS)
        var io: Operand? = null
        var wo: Operand? = null
        var bo: Operand? = null
        var colPtr = 0
        var geoPtr = 0
        try {
            io = operand(input) ?: return null
            wo = weightOperand(weight) ?: return null
            bo = bias?.let { operand(it) ?: return null }
            colPtr = WasmKernels.alloc(colBytes.toInt())
            geoPtr = WasmKernels.alloc(geo.size * 4)
            WasmKernels.writeInts(geoPtr, geo, 0, geo.size)
            val inPtr = io.ptr
            val wPtr = wo.ptr
            val bPtr = bo?.ptr ?: 0
            val cPtr = colPtr
            val gPtr = geoPtr
            val inputs = listOfNotNull(input, weight, bias).toTypedArray()
            return output(outShape, input.dtype, inputs) { out -> WasmKernels.f32Conv(inPtr, wPtr, bPtr, out, cPtr, gPtr) }
        } finally {
            if (geoPtr != 0) WasmKernels.free(geoPtr)
            if (colPtr != 0) WasmKernels.free(colPtr)
            release(bo)
            release(wo)
            release(io)
        }
    }

    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?,
        stride: Int, padding: Int, dilation: Int, groups: Int,
    ): Tensor<T, V> =
        conv(input, weight, bias, 1, intArrayOf(stride), intArrayOf(padding), intArrayOf(dilation), groups)
            ?: super.conv1d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?,
        stride: Pair<Int, Int>, padding: Pair<Int, Int>, dilation: Pair<Int, Int>, groups: Int,
    ): Tensor<T, V> =
        conv(
            input, weight, bias, 2,
            intArrayOf(stride.first, stride.second), intArrayOf(padding.first, padding.second), intArrayOf(dilation.first, dilation.second), groups,
        ) ?: super.conv2d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>, padding: Triple<Int, Int, Int>, dilation: Triple<Int, Int, Int>, groups: Int,
    ): Tensor<T, V> =
        conv(
            input, weight, bias, 3,
            intArrayOf(stride.first, stride.second, stride.third),
            intArrayOf(padding.first, padding.second, padding.third),
            intArrayOf(dilation.first, dilation.second, dilation.third),
            groups,
        ) ?: super.conv3d(input, weight, bias, stride, padding, dilation, groups)
}
