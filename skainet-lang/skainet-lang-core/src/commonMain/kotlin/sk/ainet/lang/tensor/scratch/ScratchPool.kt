package sk.ainet.lang.tensor.scratch

/**
 * Pool of reusable [FloatArray] scratch buffers, scoped to a single forward
 * (or backward) pass.
 *
 * Workspace allocation is generic across nn workloads — attention, RoPE,
 * convolutions, embedding gathers, training-time gradient buffers all need
 * short-lived intermediates. Routing those through a pool eliminates per-step
 * allocation pressure on the GC heap.
 *
 * Typical lifecycle:
 *
 *  1. The runtime owns a [ScratchPool] (one per forward-running thread).
 *  2. The runtime opens a [scope] block around each forward pass.
 *  3. Layers acquire buffers via [acquireFloat] / [acquireFloatZeroed] from
 *     the [sk.ainet.context.ExecutionContext.scratch] field.
 *  4. On scope exit every buffer acquired in the scope is recycled to the
 *     per-size-class free list for the next pass.
 *
 * Buffers may be larger than the requested size — callers must always index
 * within the size they asked for, not `buf.size`. This is the standard
 * scratch-buffer contract: external `outSize`/`n`/`stride` variables drive
 * iteration; the array's own `.size` is an implementation detail.
 *
 * Acquires made *outside* an active scope are not tracked and become regular
 * allocations (the buffer is not returned to the pool when the caller drops
 * it). This is intentional — it lets the no-op pool implementation avoid
 * any scope bookkeeping.
 *
 * Single-threaded by intent: one forward pass at a time per pool.
 * Concurrent forwards must use separate pools.
 */
public interface ScratchPool {
    /**
     * Acquire a [FloatArray] with at least [minSize] elements. Contents are
     * unspecified — callers must overwrite the range they read.
     */
    public fun acquireFloat(minSize: Int): FloatArray

    /**
     * Acquire a [FloatArray] with at least [minSize] elements, with the
     * range `[0, minSize)` zero-filled. Use when the caller relies on
     * sparse-write zeros (e.g. padding a smaller block into a larger
     * destination and leaving the gap implicit).
     */
    public fun acquireFloatZeroed(minSize: Int): FloatArray

    /**
     * Open a forward-pass scope. All buffers acquired inside [block] are
     * recycled at exit. Scopes may be nested; each open/close balances.
     */
    public fun <R> scope(block: () -> R): R

    /** Stats for benchmarking and leak detection. */
    public fun stats(): ScratchStats
}

public data class ScratchStats(
    val acquireCount: Long,
    val cacheHits: Long,
    val highWaterBytes: Long,
    val activeBuffers: Int
)

/**
 * No-op pool: every [acquireFloat] / [acquireFloatZeroed] allocates a fresh
 * array; [scope] just runs the block. Default carrier returned by
 * [sk.ainet.context.ExecutionContext.scratch] when no pooling is configured —
 * preserves pre-pool behavior bit-for-bit.
 */
public object NoopScratchPool : ScratchPool {
    override fun acquireFloat(minSize: Int): FloatArray = FloatArray(minSize)
    override fun acquireFloatZeroed(minSize: Int): FloatArray = FloatArray(minSize)
    override fun <R> scope(block: () -> R): R = block()
    override fun stats(): ScratchStats = ScratchStats(0L, 0L, 0L, 0)
}

/**
 * Size-classed slab pool with power-of-two buckets starting at 64 floats.
 *
 * Sizes round up to the next power of two: `1..64 → 64`, `65..128 → 128`,
 * ... up to [MAX_CLASSES] classes (`64 * 2^19 ≈ 33M floats` = 128 MB at the
 * top of the range). Up to [maxBuffersPerClass] buffers are retained per
 * class; surplus buffers are dropped to GC at scope exit.
 *
 * Choice rationale: hotspot sizes in nn workloads are model-derived and
 * predictable (head_dim, seq_len, n_heads). Power-of-two slabs cap
 * fragmentation at 2× per buffer with no hash-map churn — strictly better
 * than a free-list-by-exact-size for these access patterns.
 */
public class SizeClassedScratchPool(
    private val maxBuffersPerClass: Int = 8
) : ScratchPool {

    private val classes: Array<ArrayDeque<FloatArray>> =
        Array(MAX_CLASSES) { ArrayDeque() }

    /** Stack of scope frames; each frame is the list of buffers acquired in
     *  that scope. `addLast` on enter, drain on exit. */
    private val scopeStack: ArrayDeque<ArrayDeque<FloatArray>> = ArrayDeque()

    private var acquireCount: Long = 0L
    private var cacheHits: Long = 0L
    private var highWaterBytes: Long = 0L
    private var currentBytes: Long = 0L

    override fun acquireFloat(minSize: Int): FloatArray {
        val cls = sizeClass(minSize)
        val cache = classes[cls]
        acquireCount++
        val buf = if (cache.isNotEmpty()) {
            cacheHits++
            cache.removeLast()
        } else {
            FloatArray(sizeForClass(cls))
        }
        scopeStack.lastOrNull()?.addLast(buf)
        currentBytes += buf.size.toLong() * 4L
        if (currentBytes > highWaterBytes) highWaterBytes = currentBytes
        return buf
    }

    override fun acquireFloatZeroed(minSize: Int): FloatArray {
        val buf = acquireFloat(minSize)
        buf.fill(0f, 0, minSize)
        return buf
    }

    override fun <R> scope(block: () -> R): R {
        val frame: ArrayDeque<FloatArray> = ArrayDeque()
        scopeStack.addLast(frame)
        try {
            return block()
        } finally {
            scopeStack.removeLast()
            for (buf in frame) returnToCache(buf)
        }
    }

    override fun stats(): ScratchStats {
        var active = 0
        for (frame in scopeStack) active += frame.size
        return ScratchStats(acquireCount, cacheHits, highWaterBytes, active)
    }

    private fun returnToCache(buf: FloatArray) {
        currentBytes -= buf.size.toLong() * 4L
        val cls = sizeClass(buf.size)
        val cache = classes[cls]
        if (cache.size < maxBuffersPerClass) cache.addLast(buf)
    }

    public companion object {
        public const val MIN_SIZE: Int = 64
        public const val LOG_MIN_SIZE: Int = 6
        public const val MAX_CLASSES: Int = 20

        /**
         * Bucket index for a request of [minSize] floats. Returns 0 for any
         * `minSize <= MIN_SIZE`; otherwise the smallest class whose size is
         * `>= minSize`. Capped at [MAX_CLASSES] - 1 — requests beyond that
         * still allocate an array of the requested rounded size, but reuse
         * is bounded by the top class.
         */
        public fun sizeClass(minSize: Int): Int {
            if (minSize <= MIN_SIZE) return 0
            val cls = (32 - (minSize - 1).countLeadingZeroBits()) - LOG_MIN_SIZE
            return cls.coerceIn(0, MAX_CLASSES - 1)
        }

        /** Floats in bucket [cls]. */
        public fun sizeForClass(cls: Int): Int = MIN_SIZE shl cls
    }
}
