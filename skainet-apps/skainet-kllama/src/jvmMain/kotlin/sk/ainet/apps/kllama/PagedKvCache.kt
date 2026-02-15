package sk.ainet.apps.kllama

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder

/**
 * Paged KV cache backed by lazily-allocated [MemorySegment] pages.
 *
 * Instead of pre-allocating memory for the full `[nLayers * seqLen * kvDim]` grid,
 * this cache allocates fixed-size pages on first access. This dramatically reduces
 * memory usage for short sequences while still supporting long context windows.
 *
 * Page geometry:
 * - Each page covers a contiguous span of sequence positions for one layer.
 * - Pages are 64-byte aligned for SIMD-friendly access.
 * - The page grid is `[nLayers][nContextPages]` where
 *   `nContextPages = ceil(seqLen / positionsPerPage)`.
 *
 * Ported from Jlama's `KvBufferCache.computePageSize()` algorithm.
 */
public class PagedKvCache(
    override val nLayers: Int,
    override val seqLen: Int,
    override val kvDim: Int,
    pageBudgetBytes: Long = 8L * 1024 * 1024, // 8 MB default
) : KvCache, AutoCloseable {

    private val arena: Arena = Arena.ofShared()

    /** Number of sequence positions per page. */
    internal val positionsPerPage: Int

    /** Number of context pages per layer. */
    internal val nContextPages: Int

    /** Bytes per page. */
    internal val pageBytes: Long

    /** Total pages allocated so far. */
    @Volatile
    public var allocatedPages: Int = 0
        private set

    // Lazy 2D page grid: [layer][contextPage] -> MemorySegment (key) or null
    private val keyPages: Array<Array<MemorySegment?>>
    private val valuePages: Array<Array<MemorySegment?>>

    private val FLOAT_LE: ValueLayout.OfFloat =
        ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

    init {
        require(nLayers > 0 && seqLen > 0 && kvDim > 0)

        // Compute page geometry: each page stores `positionsPerPage * kvDim` floats
        val floatBytes = Float.SIZE_BYTES.toLong()
        val bytesPerPosition = kvDim * floatBytes
        positionsPerPage = maxOf(1, (pageBudgetBytes / bytesPerPosition).toInt())
        nContextPages = (seqLen + positionsPerPage - 1) / positionsPerPage
        pageBytes = positionsPerPage.toLong() * kvDim * floatBytes

        keyPages = Array(nLayers) { arrayOfNulls(nContextPages) }
        valuePages = Array(nLayers) { arrayOfNulls(nContextPages) }
    }

    // ---- page management ----

    private fun getOrAllocKeyPage(layerIdx: Int, pageIdx: Int): MemorySegment {
        var page = keyPages[layerIdx][pageIdx]
        if (page == null) {
            synchronized(this) {
                page = keyPages[layerIdx][pageIdx]
                if (page == null) {
                    page = arena.allocate(pageBytes, 64L)
                    page!!.fill(0)
                    keyPages[layerIdx][pageIdx] = page
                    allocatedPages++
                }
            }
        }
        return page!!
    }

    private fun getOrAllocValuePage(layerIdx: Int, pageIdx: Int): MemorySegment {
        var page = valuePages[layerIdx][pageIdx]
        if (page == null) {
            synchronized(this) {
                page = valuePages[layerIdx][pageIdx]
                if (page == null) {
                    page = arena.allocate(pageBytes, 64L)
                    page!!.fill(0)
                    valuePages[layerIdx][pageIdx] = page
                    allocatedPages++
                }
            }
        }
        return page!!
    }

    // ---- coordinate helpers ----

    private fun pageIdx(position: Int): Int = position / positionsPerPage
    private fun pageOffset(position: Int): Int = position % positionsPerPage
    private fun byteOffset(posInPage: Int, headOffset: Int, elementIdx: Int): Long =
        (posInPage.toLong() * kvDim + headOffset + elementIdx) * Float.SIZE_BYTES

    // ---- KvCache interface ----

    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int,
    ) {
        require(layerIdx in 0 until nLayers) { "layerIdx $layerIdx out of bounds" }
        require(position in 0 until seqLen) { "position $position out of bounds" }

        val pIdx = pageIdx(position)
        val posInPage = pageOffset(position)

        val keyPage = getOrAllocKeyPage(layerIdx, pIdx)
        val valuePage = getOrAllocValuePage(layerIdx, pIdx)

        val baseByteOffset = posInPage.toLong() * kvDim * Float.SIZE_BYTES
        MemorySegment.copy(
            keys, keysOffset,
            keyPage, FLOAT_LE, baseByteOffset,
            kvDim,
        )
        MemorySegment.copy(
            values, valuesOffset,
            valuePage, FLOAT_LE, baseByteOffset,
            kvDim,
        )
    }

    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val pIdx = pageIdx(position)
        val page = keyPages[layerIdx][pIdx] ?: return 0f
        return page.get(FLOAT_LE, byteOffset(pageOffset(position), headOffset, elementIdx))
    }

    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val pIdx = pageIdx(position)
        val page = valuePages[layerIdx][pIdx] ?: return 0f
        return page.get(FLOAT_LE, byteOffset(pageOffset(position), headOffset, elementIdx))
    }

    override fun reset() {
        for (layer in 0 until nLayers) {
            for (p in 0 until nContextPages) {
                keyPages[layer][p]?.fill(0)
                valuePages[layer][p]?.fill(0)
            }
        }
    }

    // ---- extended methods for SIMD attention ----

    /**
     * Compute attention key scores using vectorized dot product over pages.
     *
     * @param layerIdx Layer index
     * @param query Query vector
     * @param headSize Size of each head
     * @param kvHeadIdx KV head index
     * @param currentPos Current position (scores computed for 0..currentPos)
     * @param scale Attention scale (1/sqrt(headSize))
     * @param output Output scores array (length >= currentPos+1)
     */
    public fun computeKeyScores(
        layerIdx: Int,
        query: FloatArray,
        headSize: Int,
        kvHeadIdx: Int,
        currentPos: Int,
        scale: Float,
        output: FloatArray,
    ) {
        val headOffset = kvHeadIdx * headSize
        for (t in 0..currentPos) {
            val pIdx = pageIdx(t)
            val page = keyPages[layerIdx][pIdx] ?: run {
                output[t] = 0f
                continue
            }
            val off = byteOffset(pageOffset(t), headOffset, 0)
            var score = 0f
            for (i in 0 until headSize) {
                score += query[i] * page.get(FLOAT_LE, off + i.toLong() * Float.SIZE_BYTES)
            }
            output[t] = score * scale
        }
    }

    /**
     * Compute weighted sum of values for attention output.
     */
    public fun weightedValueSum(
        layerIdx: Int,
        weights: FloatArray,
        headSize: Int,
        kvHeadIdx: Int,
        currentPos: Int,
        output: FloatArray,
        outputOffset: Int = 0,
    ) {
        val headOffset = kvHeadIdx * headSize
        for (i in 0 until headSize) {
            output[outputOffset + i] = 0f
        }
        for (t in 0..currentPos) {
            val pIdx = pageIdx(t)
            val page = valuePages[layerIdx][pIdx] ?: continue
            val off = byteOffset(pageOffset(t), headOffset, 0)
            val w = weights[t]
            for (i in 0 until headSize) {
                output[outputOffset + i] += w * page.get(FLOAT_LE, off + i.toLong() * Float.SIZE_BYTES)
            }
        }
    }

    // ---- diagnostics ----

    /**
     * Returns estimated memory usage.
     */
    public fun memoryUsageBytes(): Long = allocatedPages.toLong() * pageBytes

    public fun memoryStats(): OffheapKvCache.MemoryStats = OffheapKvCache.MemoryStats(
        nLayers = nLayers,
        seqLen = seqLen,
        kvDim = kvDim,
        totalBytes = memoryUsageBytes(),
    )

    override fun close() {
        arena.close()
    }
}
