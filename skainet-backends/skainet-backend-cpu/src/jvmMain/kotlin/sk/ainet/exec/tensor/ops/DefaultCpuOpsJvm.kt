package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.VectorOperators
import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MemSegMatmulWtKernel
import sk.ainet.backend.api.kernel.Fp32AttentionKernel
import sk.ainet.backend.api.kernel.Fp32NormKernel
import sk.ainet.backend.api.kernel.Fp32SoftmaxKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.MemSegKernelProvider
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.backend.api.kernel.KernelStrictness
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarFp16MatmulKernel
import sk.ainet.exec.kernel.ScalarMatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_0MatmulKernel
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.MemorySegmentBackedData
import sk.ainet.lang.tensor.data.MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.tensor.data.Q4_0TensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.data.TensorDataFactory
import java.lang.foreign.Arena
import kotlin.math.max

internal class DefaultCpuOpsJvm(
    dataFactory: TensorDataFactory,
    schedule: sk.ainet.context.schedule.Schedule = sk.ainet.context.schedule.Schedule.Sequential,
) : DefaultCpuOpsBase(dataFactory, schedule), sk.ainet.lang.tensor.ops.FusedNormOps {

    override fun withSchedule(schedule: sk.ainet.context.schedule.Schedule): sk.ainet.lang.tensor.ops.TensorOps =
        if (schedule === this.schedule) this else DefaultCpuOpsJvm(dataFactory, schedule)

    // Read through JvmVectorKernels (an object → static final field): C2 constant-folds static
    // finals but not instance finals, and the Vector API only intrinsifies with a constant species.
    // As an instance field every FloatVector op here ran the allocating Java fallback.
    private inline val floatSpecies: VectorSpecies<Float> get() = JvmVectorKernels.floatSpecies

    /**
     * On the JVM, auto-install ServiceLoader-discovered providers (Panama Vector,
     * native FFM) so the base class's platform-neutral packed-quant dispatch
     * (`chooseQuantizedMatmulHeap`, used for Q5_1/Q5_0 and the non-JVM path) resolves
     * the SIMD/FFM kernels rather than only the scalar floor.
     */
    override fun ensureKernelProviders() {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
    }

    /**
     * FP32 matmul kernel resolved via [KernelRegistry]. First access on a
     * given instance auto-installs providers via [KernelServiceLoader]
     * if the registry is empty; subsequent calls reuse the cached
     * lookup. Apps that prefer to wire their own providers can call
     * `KernelRegistry.register(...)` before constructing this op set.
     * Falls back to [ScalarMatmulKernel] only when no provider reports
     * itself available — in practice, [PanamaVectorKernelProvider]
     * (priority 50) wins on JDK 21+ with the incubator module loaded.
     */
    private val fp32MatmulKernel: Fp32MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.bestAvailable()?.matmulFp32() ?: ScalarMatmulKernel
    }

    /**
     * BF16 matmul kernel resolved via [KernelRegistry]. Unlike the Q4_K
     * and Q8_0 lookups (nullable, with legacy `JvmQuantizedVectorKernels`
     * fallbacks), BF16 has no pre-SPI implementation in this codebase —
     * the scalar SPI kernel is the floor. We mirror [fp32MatmulKernel]'s
     * pattern: non-null, picks the highest-priority provider that carries
     * a BF16 kernel (native FFM at 100, Panama Vector at 50), falls back
     * to [ScalarBf16MatmulKernel] when no SIMD provider reports
     * availability (e.g. tests that explicitly clear the registry).
     */
    private val bf16MatmulKernel: Bf16MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulBf16() != null }
            ?.matmulBf16()
            ?: ScalarBf16MatmulKernel
    }

    /** FP16 matmul kernel resolved via [KernelRegistry]; mirrors [bf16MatmulKernel]. */
    private val fp16MatmulKernel: Fp16MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulFp16() != null }
            ?.matmulFp16()
            ?: ScalarFp16MatmulKernel
    }

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.add(y) }) { x, y -> x + y }?.let { return it }
        return super.add(a, b)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.sub(y) }) { x, y -> x - y }?.let { return it }
        return super.subtract(a, b)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.mul(y) }) { x, y -> x * y }?.let { return it }
        return super.multiply(a, b)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.div(y) }) { x, y -> x / y }?.let { return it }
        return super.divide(a, b)
    }

    /**
     * The common row-dot is not this tier's direct path: a dense weight goes to the native `x · Wᵀ`
     * kernel ([matmulDenseWeightTransposedNative]) when the FFM provider is loaded, and to the cached
     * transpose plus the vectorised `[k, n]` kernels otherwise (SKEEP-007 §T0.2).
     */
    override fun preferDirectWeightTransposed(): Boolean = false

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        // A dense FP32 `[out, in]` weight with a native direct kernel is multiplied in place, no
        // transposed copy made or cached (the weight is made resident off-heap once instead).
        if (!isHeapPackedWeight(weight.data)) {
            matmulDenseWeightTransposedNative(x, weight)?.let { return it }
        }
        // Everything else is the common path, which looks the kernel up by the key of the two
        // views (SKEEP-003, #1507). Its kernels and its reference are commonMain and cannot read a
        // MemorySegment-backed activation (SKEEP-004 forbids element access over SegmentStorage by
        // design), so such an activation is bulk-copied to the heap once per call first: a
        // decode-step activation is k floats, and the copy is trivial beside the matmul it unlocks.
        segmentActivationToHeap(x)?.let { return super.matmulWeightTransposed(it, weight) }
        return super.matmulWeightTransposed(x, weight)
    }

    /** The vectorised feed-order kernels are this tier's fast path for a packed weight (#1096). */
    override fun preferFeedOrderPackedKernels(): Boolean = true

    /**
     * A heap FP32 copy of an activation whose bytes live off-heap, or null when [x] is not that
     * shape. Two producers exist on this tier: TensorData that is itself segment-backed
     * ([MemorySegmentBackedData]), and TensorData whose VIEW presents a
     * [sk.ainet.lang.memory.SegmentStorage] (SKEEP-004 off-heap scope allocations).
     */
    private fun <T : DType, V> segmentActivationToHeap(x: Tensor<T, V>): Tensor<T, V>? {
        if (x.dtype != FP32::class) return null
        val count = x.shape.volume
        val seg = segmentFloatWindow(x.data) ?: return null
        val arr = FloatArray(count)
        java.lang.foreign.MemorySegment.copy(seg.segment, java.lang.foreign.ValueLayout.JAVA_FLOAT, seg.byteOffset, arr, 0, count)
        // NOT dataFactory.adoptFloatArray: this JVM factory adopts float arrays back INTO
        // MemorySegments — the round-trip that defeats the whole copy. The base heap class is
        // what the common dispatch path can read.
        @Suppress("UNCHECKED_CAST")
        val data = sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(x.shape, arr) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(data, x.dtype, x)
    }

    /** Relayouted weights keyed by the identity of the bytes they came from (#1096). */

    /**
     * The direct `x · Wᵀ` kernel over an off-heap weight (`skainet_fp32_matmul_wt` through FFM), from
     * the highest-priority available provider that carries one; `null` on a JVM without the native
     * library, where the cached-transpose path below still serves.
     */
    private val fp32WtKernel: Fp32MemSegMatmulWtKernel? by lazy {
        ensureKernelProviders()
        KernelRegistry.providers().asSequence()
            .filter { it.isAvailable() }
            .mapNotNull { (it as? MemSegKernelProvider)?.matmulFp32WtMemSeg() }
            .firstOrNull()
    }

    /**
     * Dense weights made resident off-heap for [fp32WtKernel], keyed by the identity of their
     * `TensorData` — the counterpart of the transposed-weight cache this path replaces, holding the
     * same bytes untransposed where a downcall can read them without a per-call copy. Bounded like
     * that cache; an evicted weight is simply copied again on its next use. Segments come from
     * `Arena.ofAuto()`, so a dropped entry is reclaimed with the tensor that owned it.
     */
    private val residentDenseWeights: MutableList<Pair<Any, java.lang.foreign.MemorySegment>> = mutableListOf()
    private val RESIDENT_DENSE_CACHE_LIMIT: Int = 256

    /**
     * `x · Wᵀ` for a dense FP32 `[out, in]` weight through the native direct kernel — the JVM-side
     * half of SKEEP-007 §T0.2: no transposed copy of the weight is made, the weight is read where it
     * lies (a segment-backed weight as is, a heap one after a single off-heap copy). `null` when the
     * kernel is not available or the operands are not dense FP32 the kernel can take, in which case
     * the caller continues to the cached transpose.
     *
     * The output columns — the weight's rows — are the [schedule]'s units (SKEEP-005): a product with
     * several activation rows (a prefill or encoder pass) runs as up to `parallelism` native calls,
     * each over a disjoint slice of the weight's rows and all activation rows, into a private
     * `[m, cols]` block copied into the result. The kernel reads its weight rows once per activation
     * row, so a slice narrow enough for a core's own cache is what makes the pass scale; a slice is
     * never wider than the kernel's own threading threshold, which keeps the C row pool out of a
     * scheduled call. Every output element is the same contiguous dot product in the same `p` order
     * whichever task computes it, so a scheduled run is bit-identical to a sequential one. A single
     * row (a decode step) has nothing to amortise and goes to the kernel as one call, whose own pool
     * threads over the columns.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> matmulDenseWeightTransposedNative(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V>? {
        if (weight.shape.rank != 2 || x.rank < 1 || x.dtype != FP32::class || weight.dtype != FP32::class) return null
        val kernel = fp32WtKernel ?: return null
        val n = weight.shape[0]
        val k = weight.shape[1]
        if (x.shape.dimensions.last() != k) return null
        val resident = residentDenseWeight(weight, n.toLong() * k) ?: return null
        val activation = heapActivation(x) ?: return null
        val m = if (k == 0) 0 else x.shape.volume / k
        val out = FloatArray(m * n)
        if (m > 0 && n > 0 && !blasWeightTransposed(activation, resident, out, m, n, k)) {
            val (xArr, xOff) = activation
            val (wSeg, wOff) = resident
            val tasks = if (m >= 2 && m.toLong() * n >= PARALLEL_MATMUL_MIN_OUTPUT && isSharedSegment(wSeg)) {
                sk.ainet.context.schedule.Schedule.tasksFor(n, WT_COLUMN_GRAIN, schedule.parallelism)
            } else {
                1
            }
            if (tasks <= 1) {
                kernel.matmulWt(xArr, xOff, wSeg, wOff, out, 0, m, n, k)
            } else {
                val grain = maxOf(WT_COLUMN_GRAIN, (n + tasks - 1) / tasks)
                schedule.forRange(n, grain) { o0, o1 ->
                    val cols = o1 - o0
                    val block = FloatArray(m * cols)
                    kernel.matmulWt(xArr, xOff, wSeg, wOff + o0.toLong() * k * java.lang.Float.BYTES, block, 0, m, cols, k)
                    for (i in 0 until m) System.arraycopy(block, i * cols, out, i * n + o0, cols)
                }
            }
        }
        val outShape = if (x.rank == 1) Shape(n) else Shape(*(x.shape.dimensions.copyOf(x.rank - 1) + n))
        val outData = dataFactory.adoptFloatArray<T, Float>(outShape, x.dtype, out) as TensorData<T, V>
        return newTensor(outData, x.dtype, x, weight)
    }

    /**
     * The activation's floats and their offset on the heap; a dense segment-backed activation is
     * copied (it is `m × k`, small). `MemorySegmentTensorData` specifically, not the marker
     * interface: the Q4/Q8 segment types implement that too and hold packed bytes, not floats.
     */
    private fun heapActivation(x: Tensor<*, *>): Pair<FloatArray, Int>? {
        heapFloatWindow(x.data)?.let { return it.arr to it.off }
        val seg = segmentFloatWindow(x.data) ?: return null
        val count = x.shape.volume
        val arr = FloatArray(count)
        java.lang.foreign.MemorySegment.copy(seg.segment, java.lang.foreign.ValueLayout.JAVA_FLOAT, seg.byteOffset, arr, 0, count)
        return arr to 0
    }

    /** A dense FP32 segment and the byte offset of element 0. */
    private class SegmentWindow(@JvmField val segment: java.lang.foreign.MemorySegment, @JvmField val byteOffset: Long)

    /**
     * The off-heap floats behind [data] when its view is dense FP32, gap-free and over a
     * [sk.ainet.lang.memory.SegmentStorage]; `null` otherwise. Read off the view, not the class
     * (SKEEP-003, #1507): the packed Q4/Q8 segment types are not dense and so never match.
     */
    private fun segmentFloatWindow(data: sk.ainet.lang.tensor.data.TensorData<*, *>): SegmentWindow? {
        val view = data.view ?: return null
        if (view.format != DENSE_FP32 || !view.isContiguous) return null
        val storage = view.storage as? sk.ainet.lang.memory.SegmentStorage ?: return null
        return SegmentWindow(storage.segment(), view.layout.offsetElements * java.lang.Float.BYTES.toLong())
    }

    /**
     * `x · Wᵀ` through the platform BLAS when one is enabled and resolved ([JvmBlas]): `sgemm` with B
     * transposed reads the resident `[out, in]` weight where it lies, the activation is staged
     * off-heap (`m × k`, small) and the `m × n` result copied back. On macOS this is Accelerate, whose
     * SGEMM runs on the matrix units of Apple silicon and threads itself — several times the NEON
     * kernel's throughput — so it takes the product before the schedule split. False when BLAS is
     * off, absent, the product is too small to pay the call, or the call failed; the caller then
     * runs the native kernel.
     */
    private fun blasWeightTransposed(
        activation: Pair<FloatArray, Int>,
        resident: Pair<java.lang.foreign.MemorySegment, Long>,
        out: FloatArray,
        m: Int, n: Int, k: Int,
    ): Boolean {
        if (!blasWanted(m.toLong() * n * k)) return false
        val (xArr, xOff) = activation
        val (wSeg, wOff) = resident
        Arena.ofConfined().use { arena ->
            val xSeg = arena.allocate(m.toLong() * k * java.lang.Float.BYTES, 64L)
            java.lang.foreign.MemorySegment.copy(xArr, xOff, xSeg, java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, m * k)
            val cSeg = arena.allocate(m.toLong() * n * java.lang.Float.BYTES, 64L)
            if (!JvmBlas.sgemm(false, true, m, n, k, xSeg, 0L, k, wSeg, wOff, k, cSeg, 0L, n)) return false
            java.lang.foreign.MemorySegment.copy(cSeg, java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, out, 0, m * n)
            return true
        }
    }

    /** Whether a dense FP32 product of [work] multiply-adds goes to the platform BLAS: enabled, resolved, and large enough to pay the call and the staging copies. */
    private fun blasWanted(work: Long): Boolean =
        work >= BLAS_MIN_WORK && JvmCpuBackendConfig.blasEnabled && JvmBlas.isAvailable()

    /**
     * Whether [segment] may be read from a thread other than the current one — a shared, automatic
     * or global arena. A confined segment (another caller's `Arena.ofConfined()`) is not, and a
     * product over it stays on the calling thread. The probe thread is never started.
     */
    private fun isSharedSegment(segment: java.lang.foreign.MemorySegment): Boolean = segment.isAccessibleBy(SEGMENT_PROBE_THREAD)

    /** The weight's `floats` floats as an off-heap segment and byte offset: its own segment, or the resident copy of its heap array. */
    private fun residentDenseWeight(weight: Tensor<*, *>, floats: Long): Pair<java.lang.foreign.MemorySegment, Long>? {
        val bytes = floats * java.lang.Float.BYTES
        segmentFloatWindow(weight.data)?.let { seg ->
            if (seg.byteOffset + bytes > seg.segment.byteSize()) return null
            return seg.segment to seg.byteOffset
        }
        // A heap weight is copied off-heap once and kept, keyed by the identity of its array.
        val win = heapFloatWindow(weight.data) ?: return null
        if (win.off != 0 || win.arr.size.toLong() < floats) return null
        val source = win.arr
        synchronized(residentDenseWeights) {
            residentDenseWeights.firstOrNull { it.first === source }?.let { return it.second to 0L }
            val seg = Arena.ofAuto().allocate(maxOf(bytes, 4L), 64L)
            if (floats > 0) {
                java.lang.foreign.MemorySegment.copy(source, 0, seg, java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, floats.toInt())
            }
            if (residentDenseWeights.size >= RESIDENT_DENSE_CACHE_LIMIT) residentDenseWeights.removeAt(0)
            residentDenseWeights.add(Pair(source, seg))
            return seg to 0L
        }
    }

    // ── fused row kernels (SKEEP-007 fused ops): softmax, the normalisations, attention ──────────

    /** The highest-priority available provider's kernel from [pick], or `null` (the common path serves). */
    private fun <K : Any> fusedKernel(pick: (KernelProvider) -> K?): K? {
        ensureKernelProviders()
        return KernelRegistry.providers().asSequence().filter { it.isAvailable() }.mapNotNull(pick).firstOrNull()
    }

    private val softmaxKernel: Fp32SoftmaxKernel? by lazy { fusedKernel { it.softmaxFp32() } }
    private val normKernel: Fp32NormKernel? by lazy { fusedKernel { it.normFp32() } }
    private val attentionKernel: Fp32AttentionKernel? by lazy { fusedKernel { it.attentionFp32() } }

    /**
     * Below this many elements a row kernel is not worth the downcall and its two staging copies:
     * the common loop over a few hundred floats finishes before the segments are allocated.
     */
    private val FUSED_MIN_ELEMENTS: Int = 1024

    /** A dense FP32 tensor's floats and the offset of element 0: the array itself, a slab window, or a copy of a segment. */
    private fun denseWindow(t: Tensor<*, *>): Pair<FloatArray, Int>? = heapActivation(t)

    /** A dense FP32 tensor as one whole array from index 0 — its own buffer when it is one, a copy otherwise. */
    private fun wholeFloats(t: Tensor<*, *>): FloatArray {
        val win = heapFloatWindow(t.data) ?: return t.data.copyToFloatArray()
        val n = t.shape.volume
        return if (win.off == 0 && win.arr.size == n) win.arr else win.arr.copyOfRange(win.off, win.off + n)
    }

    private fun <T : DType, V> softmaxNative(tensor: Tensor<T, V>, dim: Int, log: Boolean): Tensor<T, V>? {
        val kernel = softmaxKernel ?: return null
        if (tensor.dtype != FP32::class) return null
        val rank = tensor.rank
        if (rank == 0) return null
        val nd = if (dim < 0) dim + rank else dim
        if (nd != rank - 1) return null
        val cols = tensor.shape[nd]
        val n = tensor.shape.volume
        if (cols <= 0 || n < FUSED_MIN_ELEMENTS) return null
        val src = denseWindow(tensor) ?: return null
        val out = FloatArray(n)
        kernel.softmaxRows(src.first, src.second, out, 0, n / cols, cols, log)
        return floatResult(tensor.shape, tensor.dtype, out, tensor)
    }

    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> =
        softmaxNative(tensor, dim, log = false) ?: super.softmax(tensor, dim)

    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> =
        softmaxNative(tensor, dim, log = true) ?: super.logSoftmax(tensor, dim)

    /** `cols` values of a norm parameter as one array, or `null` when its size does not match. */
    private fun normParam(t: Tensor<*, *>, cols: Int): FloatArray? {
        if (t.shape.volume != cols) return null
        return wholeFloats(t)
    }

    private fun <T : DType, V> normNative(
        input: Tensor<T, V>, gamma: Tensor<T, V>?, beta: Tensor<T, V>?, eps: Float, normalizedRank: Int, rms: Boolean,
    ): Tensor<T, V>? {
        val kernel = normKernel ?: return null
        if (input.dtype != FP32::class) return null
        val rank = input.rank
        if (normalizedRank !in 1..rank) return null
        var cols = 1
        for (i in rank - normalizedRank until rank) cols *= input.shape[i]
        val n = input.shape.volume
        if (cols == 0 || n == 0) return null
        val src = denseWindow(input) ?: return null
        val g = gamma?.let { normParam(it, cols) ?: return null }
        val b = beta?.let { normParam(it, cols) ?: return null }
        val out = FloatArray(n)
        if (rms) kernel.rmsNorm(src.first, src.second, g, out, 0, n / cols, cols, eps)
        else kernel.layerNorm(src.first, src.second, g, b, out, 0, n / cols, cols, eps)
        return floatResult(input.shape, input.dtype, out, input)
    }

    override fun <T : DType, V> layerNorm(input: Tensor<T, V>, gamma: Tensor<T, V>?, beta: Tensor<T, V>?, eps: Float, normalizedRank: Int): Tensor<T, V>? =
        normNative(input, gamma, beta, eps, normalizedRank, rms = false)

    override fun <T : DType, V> rmsNorm(input: Tensor<T, V>, gamma: Tensor<T, V>?, eps: Float, normalizedRank: Int): Tensor<T, V>? =
        normNative(input, gamma, null, eps, normalizedRank, rms = true)

    /**
     * Attention through the fused native kernel when the FFM provider carries it: the same operand
     * contract and mask rules as the common implementation (which validates them here), the
     * `q · k` dot products vectorised, rows threaded in the native row pool rather than through the
     * schedule. Without the kernel, or for non-FP32 operands, the common implementation serves.
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
        val kernel = attentionKernel
        if (kernel == null || query.dtype != FP32::class || key.dtype != FP32::class || value.dtype != FP32::class) {
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
        val maskDims = sdpaMaskDims(mask, batch, heads, kvHeads, seqQ, seqKV)
        val effectiveScale = if (scale == 0f) (1.0 / kotlin.math.sqrt(headDim.toDouble())).toFloat() else scale
        val out = FloatArray(batch * heads * seqQ * headDim)
        kernel.sdpa(
            wholeFloats(query), wholeFloats(key), wholeFloats(value), out,
            batch, heads, kvHeads, seqQ, seqKV, headDim,
            effectiveScale, causal, window, rightContext,
            lengths, mask?.let { wholeFloats(it) }, maskDims,
        )
        return floatResult(Shape(batch, heads, seqQ, headDim), query.dtype, out, query, key, value)
    }

    /**
     * Below this many multiply-accumulates a dense FP32 matmul is faster done directly than handed
     * to the tile-blocked SPI kernel, whose per-call setup dominates at these sizes. Sized to cover
     * decode-step projections (m=1) while leaving prefill batches and the big vocab matmuls to the
     * kernel — at m=32 the kernel is already ~16 GFLOP/s and pulling away.
     */
    private val SMALL_FP32_MATMUL_WORK: Long = 4_000_000L

    /**
     * Below this many multiply-adds a dense FP32 product stays on the in-process kernels even when a
     * BLAS is on: the call, and on the heap faces the staging copies, cost more than the arithmetic.
     */
    private val BLAS_MIN_WORK: Long = 1L shl 18

    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        // `x · Wᵀ` written as two steps (#1108) — before everything, including the strictness check
        // below, which would otherwise report a missing kernel for an operand that has a perfectly
        // good one behind the marker.
        untransposedWeight(b)?.let { return matmulWeightTransposed(a, it) }
        // Try quantized matmul path first (FP32 input x quantized weights)
        chooseQuantizedMatmul(a, b)?.let { return it }
        // Fallback to standard FP32 matmul
        chooseMatmul(a, b)?.let { return it }
        // RFC fail-fast point: if `-Dskainet.strict.kernels=true`, surface
        // the missing kernel here rather than letting `super.matmul` quietly
        // pick the scalar dequant + FP32 fallback. The strictness check is
        // a no-op when the property is unset, preserving the existing
        // adaptive behaviour.
        KernelStrictness.failIfStrict {
            val inDt = a.dtype.simpleName ?: a.dtype.toString()
            val wDt = b.dtype.simpleName ?: b.dtype.toString()
            val providers = KernelRegistry.providers().joinToString { p ->
                "${p.name}(priority=${p.priority}, available=${p.isAvailable()})"
            }.ifEmpty { "<none>" }
            "matmul ($inDt × $wDt) has no SPI kernel; would silently fall back " +
                "to super.matmul. Registered providers: $providers"
        }
        return super.matmul(a, b)
    }

    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val rank = tensor.shape.rank
        if (rank == 2) {
            val rows = tensor.shape[0]
            val cols = tensor.shape[1]
            val data = tensor.data
            // Lazy transpose for Q4/Q8 MemorySegment data: swap shape, keep data.
            // The quantized matmul kernel accesses the segment directly with
            // (inputDim, outputDim) parameters derived from the transposed shape,
            // so physical data reordering is not needed.
            if (data is Q4MemorySegmentMarker) {
                val td = data as Q4MemorySegmentTensorData
                val transposed = Q4MemorySegmentTensorData(Shape(cols, rows), td.segment, td.segmentByteOffset)
                @Suppress("UNCHECKED_CAST")
                return newTensor(transposed as TensorData<T, V>, tensor.dtype, tensor)
            }
            if (data is Q8MemorySegmentMarker) {
                val td = data as Q8MemorySegmentTensorData
                val transposed = Q8MemorySegmentTensorData(Shape(cols, rows), td.segment, td.segmentByteOffset)
                @Suppress("UNCHECKED_CAST")
                return newTensor(transposed as TensorData<T, V>, tensor.dtype, tensor)
            }
            // Q4_K (ByteArray-backed `Q4_KBlockTensorData`) transpose used to be
            // intercepted here with a bare shape swap, on the claim that
            // `JvmQuantizedVectorKernels.matmulQ4_KVec`'s `(blockIdx * outputDim + o)`
            // byte addressing matches the packed layout "as-is". That's only true
            // when there's a single quant block per row; for any wider row it silently
            // fed the kernel physically wrong bytes — the same root cause as the
            // Q5_0/Q5_1/Q8_0/Q4_0/Q5_K/Q6_K bug fixed in `DefaultCpuOpsBase.transpose`
            // (see `transposePackedBlocks`, SKaiNET-transformers#307). Falling through
            // to `super.transpose()` picks up the corrected, physically-reordering
            // implementation instead of duplicating (and re-diverging from) it here.
            //
            // Narrow-float input-major lazy transpose is handled in DefaultCpuOpsBase too —
            // nothing above intercepts it, so it falls through. Issue #888.
            // Q6_K / Q5_1 / Q5_0 / Q4_K / Q8_0 / Q4_0 transpose is handled in
            // DefaultCpuOpsBase (block-major, shared with Native); the JVM ops don't
            // intercept them here.
            // MemorySegment FP32 fast path: physical transpose via SIMD.
            // Uses Arena.ofAuto() so the result segment is reclaimed by GC
            // when the wrapping Tensor is no longer reachable. Earlier
            // ofConfined() builds leaked an arena per call, blowing 32+ GiB
            // of direct memory in inference loops (every layer × every
            // forward pass).
            if (data is MemorySegmentBackedData) {
                val arena = Arena.ofAuto()
                val result = MemorySegmentTensorData<T>(Shape(cols, rows), arena)
                val src = data as MemorySegmentBackedData
                val floatLayout = java.lang.foreign.ValueLayout.JAVA_FLOAT
                // Bulk-load source into FloatArray, transpose via tight scalar
                // loop (JIT auto-vectorizes), bulk-write destination. Replaces
                // O(rows*cols) per-element VarHandle.get/set which dominated
                // attention-path transposes.
                val srcArr = FloatArray(rows * cols)
                java.lang.foreign.MemorySegment.copy(src.segment, floatLayout, src.segmentByteOffset, srcArr, 0, rows * cols)
                val dstArr = FloatArray(rows * cols)
                for (r in 0 until rows) {
                    val rowBase = r * cols
                    for (c in 0 until cols) {
                        dstArr[c * rows + r] = srcArr[rowBase + c]
                    }
                }
                java.lang.foreign.MemorySegment.copy(dstArr, 0, result.segment, floatLayout, result.segmentByteOffset, rows * cols)
                @Suppress("UNCHECKED_CAST")
                return newTensor(result as TensorData<T, V>, tensor.dtype, tensor)
            }
        }
        return super.transpose(tensor)
    }

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Try vectorized path for FP32
        chooseConv2d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv2d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null

        val inputFloats = floatsFromZero(input.data) ?: return null
        val weightFloats = floatsFromZero(weight.data) ?: return null
        // A bias the kernel cannot read sends the whole call to the generic path; it used to be dropped.
        val biasFloats = if (bias == null) null else floatsFromZero(bias.data) ?: return null

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]

        val cOut = weight.shape[0]
        val kH = weight.shape[2]
        val kW = weight.shape[3]

        val (sH, sW) = stride
        val (pH, pW) = padding
        val (dH, dW) = dilation

        val outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
        val outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

        if (outH <= 0 || outW <= 0) return null

        val outBuffer = FloatArray(n * cOut * outH * outW)
        val biasBuffer = biasFloats

        when {
            // Grouped convolution (includes depthwise when groups == cIn == cOut)
            groups > 1 -> {
                JvmVectorKernels.conv2dGrouped(
                    inputFloats, weightFloats, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW, dH, dW,
                    groups,
                    outH, outW,
                    outBuffer
                )
            }
            // 1x1 convolution without dilation - use optimized kernel
            kH == 1 && kW == 1 && sH == 1 && sW == 1 && pH == 0 && pW == 0 && dH == 1 && dW == 1 -> {
                JvmVectorKernels.conv2d1x1Optimized(
                    inputFloats, weightFloats, biasBuffer,
                    n, cIn, inH, inW, cOut,
                    outBuffer
                )
            }
            // Standard convolution with dilation
            dH != 1 || dW != 1 -> {
                JvmVectorKernels.conv2dIm2ColDilated(
                    inputFloats, weightFloats, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW, dH, dW,
                    outH, outW,
                    outBuffer
                )
            }
            // Standard convolution without dilation
            else -> {
                JvmVectorKernels.conv2dIm2Col(
                    inputFloats, weightFloats, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW,
                    outH, outW,
                    outBuffer
                )
            }
        }

        return floatResult(Shape(n, cOut, outH, outW), input.dtype, outBuffer)
    }

    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        chooseConv1d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv1d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null
        if (groups != 1) return null  // TODO: Add grouped conv1d support

        val inputFloats = floatsFromZero(input.data) ?: return null
        val weightFloats = floatsFromZero(weight.data) ?: return null
        // A bias the kernel cannot read sends the whole call to the generic path; it used to be dropped.
        val biasFloats = if (bias == null) null else floatsFromZero(bias.data) ?: return null

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inL = input.shape[2]

        val cOut = weight.shape[0]
        val kL = weight.shape[2]

        val outL = (inL + 2 * padding - dilation * (kL - 1) - 1) / stride + 1
        if (outL <= 0) return null

        val outBuffer = FloatArray(n * cOut * outL)

        JvmVectorKernels.conv1dIm2Col(
            inputFloats, weightFloats, biasFloats,
            n, cIn, inL,
            cOut, kL,
            stride, padding, dilation,
            outL,
            outBuffer
        )

        return floatResult(Shape(n, cOut, outL), input.dtype, outBuffer)
    }

    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        chooseConv3d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv3d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null
        if (groups != 1) return null  // TODO: Add grouped conv3d support

        val inputFloats = floatsFromZero(input.data) ?: return null
        val weightFloats = floatsFromZero(weight.data) ?: return null
        // A bias the kernel cannot read sends the whole call to the generic path; it used to be dropped.
        val biasFloats = if (bias == null) null else floatsFromZero(bias.data) ?: return null

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inD = input.shape[2]
        val inH = input.shape[3]
        val inW = input.shape[4]

        val cOut = weight.shape[0]
        val kD = weight.shape[2]
        val kH = weight.shape[3]
        val kW = weight.shape[4]

        val (sD, sH, sW) = stride
        val (pD, pH, pW) = padding
        val (dD, dH, dW) = dilation

        val outD = (inD + 2 * pD - dD * (kD - 1) - 1) / sD + 1
        val outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
        val outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

        if (outD <= 0 || outH <= 0 || outW <= 0) return null

        val outBuffer = FloatArray(n * cOut * outD * outH * outW)

        JvmVectorKernels.conv3dIm2Col(
            inputFloats, weightFloats, biasFloats,
            n, cIn, inD, inH, inW,
            cOut, kD, kH, kW,
            sD, sH, sW,
            pD, pH, pW,
            dD, dH, dW,
            outD, outH, outW,
            outBuffer
        )

        return floatResult(Shape(n, cOut, outD, outH, outW), input.dtype, outBuffer)
    }

    /**
     * Dispatch to vectorized quantized matmul kernels for Q8_0 and Q4_K weights.
     * Input must be FP32, weights can be Q8_0TensorData or Q4_KTensorData.
     * Also supports MemorySegment-backed quantized tensor data.
     */
    private fun <T : DType, V> chooseQuantizedMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        // Input must be FP32
        if (a.dtype != FP32::class) return null
        if (b.shape.rank != 2) return null
        if (a.shape.rank < 1) return null
        // The heap block formats: the kernel registered for the weight's format, looked up by key
        // in the common code (SKEEP-003, #1507). What is left below is narrow floats and the
        // MemorySegment weights, which do not have a view yet.
        chooseQuantizedMatmulHeap(a, b)?.let { return it }
        if (a.shape.rank == 2) return chooseQuantizedMatmul2D(a, b)

        // Defensive: attention linear projections can in principle pass `[..., in]` (see
        // linearProject's kdoc) — flatten any leading batch/sequence dims into one so the
        // specialized quant kernels below (which only understand `[batch, in]`) still get used.
        // Also covers rank-1 activations: a single-token hidden-state vector during incremental
        // decode (post-prefill, once the KV cache is warm) — `leading` is then empty and
        // `flatBatch` is 1, so `[in]` promotes to `[1, in]` and the result squeezes back down to
        // `[out]`, same as `rank > 2` already did. Previously excluded by `a.shape.rank < 2`,
        // which sent every such decode step straight to matmulGeneric's untyped per-element
        // `TensorData.get()` path — for a pre-transposed packed-quant weight (e.g.
        // PreTransposedQ4_K) that returns the raw packed byte, not a dequantized Float.
        val leading = a.shape.dimensions.copyOf(a.shape.rank - 1)
        val flatBatch = leading.fold(1) { acc, d -> acc * d }
        val inputDim = a.shape.dimensions.last()
        val a2d = reshape(a, Shape(intArrayOf(flatBatch, inputDim)))
        val result2d = chooseQuantizedMatmul2D(a2d, b) ?: return null
        val outputDim = result2d.shape.dimensions.last()
        return reshape(result2d, Shape(leading + outputDim))
    }

    private fun <T : DType, V> chooseQuantizedMatmul2D(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        val bData = b.data
        val bShape = b.shape

        val batchSize = a.shape[0]
        val inputDim = a.shape[1]
        val weightInputDim = bShape[0]
        val outputDim = bShape[1]

        if (inputDim != weightInputDim) return null

        // Fast path: MemorySegment input × Q8 MemorySegment weights — avoid heap copy entirely
        val aData = a.data
        if (aData is MemorySegmentBackedData && bData is Q8MemorySegmentMarker) {
            return chooseQ8MemSegInputMatmul(aData, bData, batchSize, inputDim, outputDim, a)
        }

        // The activation as one FloatArray: its own array when it is heap floats from index 0,
        // otherwise a copy. Any other *dense* FP32 activation — segment-backed, a slab window from
        // a ScopedExecutionContext forward (#1145/#1146), a view-only wrapper — copies out through
        // its own (offset-aware) copyToFloatArray. Bailing out here instead used to drop these
        // activations to matmulGeneric, whose per-element get() on a quantized weight returns raw
        // codes, not values — silently wrong logits, the #993 class of bug. A packed/encoded
        // *activation* (encoding != null) still bails: this chooser's kernels want dense FP32 input.
        val inputBuffer: FloatArray = floatsFromZero(aData)
            ?: if (aData.encoding == null) aData.copyToFloatArray() else return null

        return when (bData) {
            is NarrowFloatTensorData -> {
                // Narrow floats are dense (not block-quantized) and the kernel SPI is a
                // full SGEMM with `(m, n, k)` strides — no per-batch loop needed,
                // unlike the matvec-shaped Q4_K / Q8_0 / Q6_K branches.
                //
                // The codec selects the kernel: the two formats have different bit layouts,
                // so decoding F16 bytes with the BF16 kernel would yield silent garbage.
                val kernel = when (bData.codec) {
                    Bf16Codec -> bf16MatmulKernel
                    Fp16Codec -> fp16MatmulKernel
                    else -> null
                }
                if (kernel == null) {
                    null
                } else {
                    val outBuffer = FloatArray(batchSize * outputDim)
                    kernel.matmul(
                        inputBuffer, 0, inputDim,
                        bData.packedData, 0, outputDim * NarrowFloatTensorData.BYTES_PER_ELEMENT,
                        outBuffer, 0, outputDim,
                        batchSize, outputDim, inputDim,
                    )
                    floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
                }
            }
            // Q6_K / Q5_1 / Q5_0 dispatch is handled in DefaultCpuOpsBase via the kernel
            // registry (block-major, shared with Native); not intercepted here.
            // MemorySegment-backed quantized weights (Q4/Q8) — dispatch to MemorySegment kernels
            is MemorySegmentBackedData -> {
                chooseQuantizedMatmulMemSeg(inputBuffer, bData, batchSize, inputDim, outputDim, a)
            }
            else -> null
        }
    }

    /**
     * Dispatch quantized matmul when weight data is MemorySegment-backed.
     * The concrete type is checked via marker interfaces set during weight loading.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> chooseQuantizedMatmulMemSeg(
        inputBuffer: FloatArray,
        bData: MemorySegmentBackedData,
        batchSize: Int,
        inputDim: Int,
        outputDim: Int,
        a: Tensor<T, V>,
    ): Tensor<T, V>? {
        val outBuffer = FloatArray(batchSize * outputDim)
        when (bData) {
            is Q4MemorySegmentMarker -> {
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    JvmQuantizedVectorKernels.matmulF32Q4_0MemSeg(
                        batchInput,
                        bData.segment,
                        bData.segmentByteOffset,
                        inputDim,
                        outputDim,
                        outBuffer,
                        batch * outputDim,
                    )
                }
            }
            is Q8MemorySegmentMarker -> {
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    JvmQuantizedVectorKernels.matmulF32Q8_0MemSeg(
                        batchInput,
                        bData.segment,
                        bData.segmentByteOffset,
                        inputDim,
                        outputDim,
                        outBuffer,
                        batch * outputDim,
                    )
                }
            }
            else -> return null
        }
        return floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
    }

    /**
     * MemorySegment input × Q8 MemorySegment weights — zero-copy path that avoids
     * materializing input as a heap FloatArray.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> chooseQ8MemSegInputMatmul(
        aData: MemorySegmentBackedData,
        bData: Q8MemorySegmentMarker,
        batchSize: Int,
        inputDim: Int,
        outputDim: Int,
        a: Tensor<T, V>,
    ): Tensor<T, V> {
        val outBuffer = FloatArray(batchSize * outputDim)
        for (batch in 0 until batchSize) {
            val inputByteOffset = aData.segmentByteOffset + batch.toLong() * inputDim * 4
            JvmQuantizedVectorKernels.matmulF32Q8_0MemSegInput(
                aData.segment,
                inputByteOffset,
                bData.segment,
                bData.segmentByteOffset,
                inputDim,
                outputDim,
                outBuffer,
                batch * outputDim,
            )
        }
        return floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        vectorFloatUnary(tensor, { vector ->
            val zero = FloatVector.zero(floatSpecies)
            vector.max(zero)
        }, { value ->
            if (value < 0f) 0f else value
        })?.let { return it }
        return super.relu(tensor)
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        val win = floatWinOf(tensor.data) ?: return super.silu(tensor)
        val buf = win.arr
        val off = win.off
        val n = tensor.shape.volume
        val out = FloatArray(n)
        for (i in 0 until n) {
            val x = buf[off + i]
            out[i] = x / (1f + kotlin.math.exp(-x))
        }
        return floatResult(Shape(tensor.shape.dimensions.copyOf()), tensor.dtype, out)
    }

    override fun <T : DType, V> erf(tensor: Tensor<T, V>): Tensor<T, V> {
        vectorFloatUnary(tensor, { v -> erfVector(v) }, ErfKernel::erfFloat)?.let { return it }
        return super.erf(tensor)
    }

    override fun <T : DType, V> gelu(tensor: Tensor<T, V>, approximate: Boolean): Tensor<T, V> {
        if (approximate) return super.gelu(tensor, true)
        vectorFloatUnary(tensor, { v ->
            val half = FloatVector.broadcast(floatSpecies, 0.5f)
            val one = FloatVector.broadcast(floatSpecies, 1f)
            half.mul(v).mul(one.add(erfVector(v.mul(ErfKernel.SQRT1_2_F))))
        }, ErfKernel::geluExactFloat)?.let { return it }
        return super.gelu(tensor, false)
    }

    /**
     * Lane-wise Abramowitz–Stegun 7.1.26 erf in single precision — the same operation order as
     * [ErfKernel.erfFloat], which serves the scalar tail, so lanes and tail agree up to the
     * last-ulp difference between `VectorOperators.EXP` and `kotlin.math.exp`.
     */
    private fun erfVector(x: FloatVector): FloatVector {
        val one = FloatVector.broadcast(floatSpecies, 1f)
        val a = x.abs()
        val t = one.div(a.mul(ErfKernel.P_F).add(1f))
        var poly = t.mul(ErfKernel.A5_F).add(ErfKernel.A4_F)
        poly = poly.mul(t).add(ErfKernel.A3_F)
        poly = poly.mul(t).add(ErfKernel.A2_F)
        poly = poly.mul(t).add(ErfKernel.A1_F)
        val e = a.mul(a).neg().lanewise(VectorOperators.EXP)
        val y = one.sub(poly.mul(t).mul(e))
        return y.blend(y.neg(), x.lt(0f))
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            vectorFloatReduceAllSum<T, V>(tensor)?.let { return it }
        }
        return super.sum(tensor, dim)
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            val volume = tensor.shape.volume
            if (volume == 0) return super.mean(tensor, dim)
            vectorFloatReduceAllSum<T, V>(tensor)?.let { reduced ->
                val out = reduced as CpuTensor<T, V>
                @Suppress("UNCHECKED_CAST")
                val sum = heapFloatWindow(out.data) ?: return super.mean(tensor, dim)
                sum.arr[sum.off] = sum.arr[sum.off] / volume.toFloat()
                return reduced
            }
        }
        return super.mean(tensor, dim)
    }

    // inline for the same reason as JvmVectorKernels.binaryFloat: the op lambdas must reach
    // the vector loops as code, not as Function2 objects, or the Vector API is not intrinsified.
    private inline fun <T : DType, V> vectorFloatBinary(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        vectorOp: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float
    ): Tensor<T, V>? {
        // Only FP32/FP16 supported for vector path
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null

        val aWin = floatWinOf(a.data) ?: return null
        val bWin = floatWinOf(b.data) ?: return null
        val aBuf = aWin.arr
        val aBase = aWin.off
        val bBuf = bWin.arr
        val bBase = bWin.off

        // Determine broadcasted output shape
        val outShape = try { broadcastShapes(a.shape, b.shape) } catch (e: IllegalArgumentException) { return null }
        val outVolume = outShape.volume
        val outBuffer = FloatArray(outVolume)

        val aVol = a.shape.volume
        val bVol = b.shape.volume

        // Case 1: exact shape match (fast path)
        if (a.shape == b.shape) {
            JvmVectorKernels.binaryFloat(aBuf, bBuf, outBuffer, outVolume, vectorOp, scalarOp, aOffset = aBase, bOffset = bBase)
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }

        // Case 2: scalar broadcast
        if (aVol == 1) {
            JvmVectorKernels.binaryFloatScalarLeft(aBuf[aBase], bBuf, bBase, outBuffer, outVolume, vectorOp, scalarOp)
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }
        if (bVol == 1) {
            JvmVectorKernels.binaryFloatScalarRight(aBuf, aBase, bBuf[bBase], outBuffer, outVolume, vectorOp, scalarOp)
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }

        // Case 3: last-dimension broadcasting (bias add). Supports arbitrary leading dims.
        val aLast = if (a.shape.rank > 0) a.shape[a.shape.rank - 1] else 1
        val bLast = if (b.shape.rank > 0) b.shape[b.shape.rank - 1] else 1
        val outLast = outShape.dimensions.lastOrNull() ?: 1
        val groups = if (outLast == 0) 0 else outVolume / outLast
        if (groups > 0) {
            // b broadcasts across leading dims if its last dim == outLast and all other dims are 1 or equal
            val bIsBias = (b.shape.rank == 1 && bLast == outLast) || (
                b.shape.rank >= 1 && bLast == outLast && b.shape.dimensions.dropLast(1).all { it == 1 }
            )
            val aIsBias = (a.shape.rank == 1 && aLast == outLast) || (
                a.shape.rank >= 1 && aLast == outLast && a.shape.dimensions.dropLast(1).all { it == 1 }
            )
            if (bIsBias && aVol == outVolume) {
                for (g in 0 until groups) {
                    val off = g * outLast
                    JvmVectorKernels.binaryFloat(
                        aBuf, bBuf, outBuffer, outLast, vectorOp, scalarOp,
                        aOffset = aBase + off, bOffset = bBase, outOffset = off,
                    )
                }
                return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
            }
            if (aIsBias && bVol == outVolume) {
                for (g in 0 until groups) {
                    val off = g * outLast
                    JvmVectorKernels.binaryFloat(
                        aBuf, bBuf, outBuffer, outLast, vectorOp, scalarOp,
                        aOffset = aBase, bOffset = bBase + off, outOffset = off,
                    )
                }
                return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
            }
        }

        // Fallback when complex broadcasting not supported here
        return null
    }

    private inline fun <T : DType, V> vectorFloatUnary(
        tensor: Tensor<T, V>,
        vectorOp: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float
    ): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val win = floatWinOf(tensor.data) ?: return null
        val volume = tensor.shape.volume
        val outBuffer = FloatArray(volume)
        JvmVectorKernels.unaryFloat(win.arr, outBuffer, volume, vectorOp, scalarOp, inputOffset = win.off)
        return floatResult(Shape(tensor.shape.dimensions.copyOf()), tensor.dtype, outBuffer)
    }

    /** Dense-FP32 window (array + base offset): plain array data at 0, slab-backed data (#1173) at its arrayOffset. */
    private fun floatWinOf(d: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatWindow? = heapFloatWindow(d)

    private fun <T : DType> supportsFloatOps(a: Tensor<T, *>, b: Tensor<T, *>): Boolean {
        return supportsFloatOps(a) &&
            a.dtype == b.dtype &&
            a.shape == b.shape
    }

    private fun <T : DType> supportsFloatOps(tensor: Tensor<T, *>): Boolean {
        val dtype = tensor.dtype
        // BF16 was excluded here, which left BF16-tagged tensors matmul-only — every elementwise
        // and unary op fell through to the generic scalar path. All three float tags are backed by
        // FloatArray when produced by DenseTensorDataFactory, so they can use the vector kernels.
        // Callers re-check with `data as? FloatArrayTensorData`, so genuinely narrow-storage
        // tensors (Bf16/Fp16DenseTensorData) still fall through safely rather than being misread.
        return dtype == FP32::class || dtype == FP16::class || dtype == BF16::class
    }

    /** `a · b` for two dense FP32 operands that both live in memory segments; `null` when either does not. */
    private fun <T : DType, V> chooseMatmulSegments(a: Tensor<T, V>, b: Tensor<T, V>, m: Int, n: Int, k: Int): Tensor<T, V>? {
        val aMemSeg = segmentFloatWindow(a.data)
        val bMemSeg = segmentFloatWindow(b.data)
        if (aMemSeg != null && bMemSeg != null) {
            // Same fix as the transpose path above: use Arena.ofAuto so the
            // matmul output segment is GC-reclaimable. Per-call ofConfined()
            // leaks ~tens of MB per matmul, which over a 35-layer Gemma 4
            // forward pass exhausts the JVM direct-memory cap.
            val arena = Arena.ofAuto()
            val result = MemorySegmentTensorData<T>(Shape(m, n), arena)
            // Both operands already off-heap: the platform BLAS reads them in place, no staging.
            if (blasWanted(m.toLong() * n * k) &&
                JvmBlas.sgemm(
                    false, false, m, n, k,
                    aMemSeg.segment, aMemSeg.byteOffset, k,
                    bMemSeg.segment, bMemSeg.byteOffset, n,
                    result.segment, result.segmentByteOffset, n,
                )
            ) {
                @Suppress("UNCHECKED_CAST")
                return CpuTensor(result as TensorData<T, V>, this, a.dtype)
            }
            val blockedThresholdMS = 16 * 16
            if (m >= blockedThresholdMS || n >= blockedThresholdMS || k >= blockedThresholdMS) {
                JvmVectorKernels.matmulFloatBlockedMemSeg(
                    m, k, n,
                    aMemSeg.segment, aMemSeg.byteOffset,
                    bMemSeg.segment, bMemSeg.byteOffset,
                    result.segment, result.segmentByteOffset,
                    schedule = schedule,
                )
            } else {
                JvmVectorKernels.matmulFloatMemSeg(
                    m, k, n,
                    aMemSeg.segment, aMemSeg.byteOffset,
                    bMemSeg.segment, bMemSeg.byteOffset,
                    result.segment, result.segmentByteOffset,
                )
            }
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(result as TensorData<T, V>, this, a.dtype)
        }
        return null
    }

    private fun <T : DType, V> chooseMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null
        if (a.shape.rank != 2 || b.shape.rank != 2) return null

        val aRows = a.shape[0]
        val aCols = a.shape[1]
        val bRows = b.shape[0]
        val bCols = b.shape[1]
        if (aCols != bRows) return null

        val m = aRows
        val n = bCols
        val k = aCols

        // ---- FloatArray path (plain arrays at offset 0, slab windows at their base — #1173) ----
        // Heap data answers without building a view, so it is asked first; only operands that are
        // not heap floats go on to the segment path, which reads their views.
        val aWin = floatWinOf(a.data)
        val bWin = floatWinOf(b.data)
        if (aWin == null || bWin == null) return chooseMatmulSegments(a, b, m, n, k)

        val work = m.toLong() * n.toLong() * k.toLong()
        val outBuffer = FloatArray(m * n)

        // The platform BLAS first when it is on and resolved (Accelerate on macOS by default): the
        // heap face stages the operands off-heap, which the native heap kernel below does too.
        if (blasWanted(work) &&
            JvmBlas.sgemmRowMajorNN(m, n, k, aWin.arr, aWin.off, k, bWin.arr, bWin.off, n, outBuffer, 0, n)
        ) {
            return floatResult(Shape(m, n), a.dtype, outBuffer)
        }

        // Small shapes are overhead-bound, not throughput-bound. The tile-blocked SPI kernel below
        // costs ~1ms per call whether it multiplies 1 row or 8 (Fp32GemvShapeBench: m=1 1.00ms,
        // m=8 1.08ms — 8x the arithmetic for 8% more time), so at decode sizes essentially all of
        // that is fixed cost. A decode step is m=1 by construction, and a model whose checkpoint
        // ships dense FP32 tensors does this per layer: Gemma 4 E2B has 70 such projections per
        // token and spent 73% of decode here. Do the small ones directly instead — contiguous in
        // `b` and `out`, one pass, no setup — and leave the tiled kernel the large shapes it wins.
        if (work <= SMALL_FP32_MATMUL_WORK) {
            val aBuf = aWin.arr
            val aBase = aWin.off
            val bBuf = bWin.arr
            val bBase = bWin.off
            for (i in 0 until m) {
                val aOff = aBase + i * k
                val outOff = i * n
                for (p in 0 until k) {
                    val av = aBuf[aOff + p]
                    if (av == 0f) continue
                    val bOff = bBase + p * n
                    for (j in 0 until n) {
                        outBuffer[outOff + j] += av * bBuf[bOff + j]
                    }
                }
            }
            return floatResult(Shape(m, n), a.dtype, outBuffer)
        }

        // Route through the kernel SPI — the registered provider
        // (Panama on JDK 21+, scalar otherwise) is tile-blocked and
        // handles small + large inputs in one path, so the previous
        // simple-vs-blocked fork is no longer needed.
        fp32MatmulKernel.matmul(
            aWin.arr, aWin.off, k,
            bWin.arr, bWin.off, n,
            outBuffer, 0, n,
            m, n, k,
        )
        return floatResult(Shape(m, n), a.dtype, outBuffer)
    }

    private fun <T : DType, V> vectorFloatReduceAllSum(tensor: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val win = floatWinOf(tensor.data) ?: return null
        val buffer = win.arr
        val base = win.off
        val n = tensor.shape.volume
        if (n == 0) return null
        // NOTE: For numerical reproducibility with Kotlin's FloatArray.sum(),
        // perform strict left-to-right scalar accumulation.
        var acc = 0.0f
        var idx = 0
        while (idx < n) {
            acc += buffer[base + idx]
            idx++
        }
        val outData = DenseFloatArrayTensorData<T>(Shape(), floatArrayOf(acc))
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, tensor.dtype)
    }
}

/** A thread that never runs: the probe for [java.lang.foreign.MemorySegment.isAccessibleBy] (a confined segment answers `false` for it). */
private val SEGMENT_PROBE_THREAD: Thread = Thread { }

/**
 * Fewest weight rows one scheduled `x · Wᵀ` task takes: four is the kernel's row tile, sixteen
 * keeps the per-call staging (an activation copy, an arena) a rounding error beside the dot products.
 */
private const val WT_COLUMN_GRAIN: Int = 16
