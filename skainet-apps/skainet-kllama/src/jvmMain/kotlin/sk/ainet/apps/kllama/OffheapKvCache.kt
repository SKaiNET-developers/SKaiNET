package sk.ainet.apps.kllama

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Off-heap KV cache implementation using direct ByteBuffers.
 *
 * This implementation stores key and value tensors in native memory outside the JVM heap,
 * reducing GC pressure for large context windows. Direct buffers also enable potential
 * future integration with native BLAS libraries.
 *
 * Memory layout (contiguous per layer):
 * - Keys: [nLayers][seqLen][kvDim] as float32
 * - Values: [nLayers][seqLen][kvDim] as float32
 *
 * @param nLayers Number of transformer layers
 * @param seqLen Maximum sequence length (context window)
 * @param kvDim KV head dimension (nKvHeads * headSize)
 */
public class OffheapKvCache(
    override val nLayers: Int,
    override val seqLen: Int,
    override val kvDim: Int
) : KvCache, AutoCloseable {

    private val floatsPerLayer = seqLen * kvDim
    private val bytesPerLayer = floatsPerLayer * Float.SIZE_BYTES
    private val totalBytes = nLayers.toLong() * bytesPerLayer

    // Allocate direct buffers for keys and values
    private val keyBuffer: ByteBuffer = ByteBuffer.allocateDirect(totalBytes.toInt())
        .order(ByteOrder.nativeOrder())
    private val valueBuffer: ByteBuffer = ByteBuffer.allocateDirect(totalBytes.toInt())
        .order(ByteOrder.nativeOrder())

    // Float views for efficient access
    private val keyFloatBuffer: FloatBuffer = keyBuffer.asFloatBuffer()
    private val valueFloatBuffer: FloatBuffer = valueBuffer.asFloatBuffer()

    init {
        require(nLayers > 0) { "nLayers must be positive" }
        require(seqLen > 0) { "seqLen must be positive" }
        require(kvDim > 0) { "kvDim must be positive" }

        // Verify allocation didn't exceed int bounds
        require(totalBytes <= Int.MAX_VALUE) {
            "Total cache size $totalBytes bytes exceeds maximum ByteBuffer size"
        }
    }

    /**
     * Store key and value vectors for a given layer and position.
     *
     * @param layerIdx Layer index (0 to nLayers-1)
     * @param position Sequence position (0 to seqLen-1)
     * @param keys Key vector of length kvDim
     * @param values Value vector of length kvDim
     */
    public fun store(layerIdx: Int, position: Int, keys: FloatArray, values: FloatArray) {
        require(layerIdx in 0 until nLayers) { "layerIdx $layerIdx out of bounds" }
        require(position in 0 until seqLen) { "position $position out of bounds" }
        require(keys.size == kvDim) { "keys size ${keys.size} != kvDim $kvDim" }
        require(values.size == kvDim) { "values size ${values.size} != kvDim $kvDim" }

        val offset = (layerIdx * seqLen + position) * kvDim
        keyFloatBuffer.position(offset)
        keyFloatBuffer.put(keys)
        valueFloatBuffer.position(offset)
        valueFloatBuffer.put(values)
    }

    /**
     * Store key and value vectors from existing arrays at given offsets.
     * Optimized for copying from existing buffers.
     *
     * @param layerIdx Layer index
     * @param position Sequence position
     * @param keys Source key array
     * @param keysOffset Starting offset in keys array
     * @param values Source value array
     * @param valuesOffset Starting offset in values array
     */
    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    ) {
        require(layerIdx in 0 until nLayers) { "layerIdx $layerIdx out of bounds" }
        require(position in 0 until seqLen) { "position $position out of bounds" }

        val offset = (layerIdx * seqLen + position) * kvDim
        keyFloatBuffer.position(offset)
        keyFloatBuffer.put(keys, keysOffset, kvDim)
        valueFloatBuffer.position(offset)
        valueFloatBuffer.put(values, valuesOffset, kvDim)
    }

    /**
     * Compute attention scores: query · keys[0:pos+1] for a single attention head.
     *
     * For GQA support, this computes scores against a specific KV head.
     *
     * @param layerIdx Layer index
     * @param query Query vector of length headSize
     * @param headSize Size of each attention head
     * @param kvHeadIdx Which KV head to attend to
     * @param currentPos Current position (computes scores for positions 0..currentPos)
     * @param scale Attention scale factor (typically 1/sqrt(headSize))
     * @param output Output array to receive scores (length >= currentPos+1)
     */
    public fun computeKeyScores(
        layerIdx: Int,
        query: FloatArray,
        headSize: Int,
        kvHeadIdx: Int,
        currentPos: Int,
        scale: Float,
        output: FloatArray
    ) {
        require(currentPos < seqLen) { "currentPos $currentPos >= seqLen $seqLen" }
        require(output.size > currentPos) { "output array too small" }

        val layerBase = layerIdx * seqLen * kvDim
        val headOffset = kvHeadIdx * headSize

        for (t in 0..currentPos) {
            val cacheBase = layerBase + t * kvDim + headOffset
            var score = 0f

            // Dot product: query · key[t]
            for (i in 0 until headSize) {
                score += query[i] * keyFloatBuffer[cacheBase + i]
            }

            output[t] = score * scale
        }
    }

    /**
     * Compute weighted sum of values: sum(weights[t] * values[t]) for t in 0..pos.
     *
     * @param layerIdx Layer index
     * @param weights Attention weights (after softmax), length >= currentPos+1
     * @param headSize Size of each attention head
     * @param kvHeadIdx Which KV head to use
     * @param currentPos Current position
     * @param output Output array of length headSize to accumulate result
     * @param outputOffset Starting offset in output array
     */
    public fun weightedValueSum(
        layerIdx: Int,
        weights: FloatArray,
        headSize: Int,
        kvHeadIdx: Int,
        currentPos: Int,
        output: FloatArray,
        outputOffset: Int = 0
    ) {
        val layerBase = layerIdx * seqLen * kvDim
        val headOffset = kvHeadIdx * headSize

        // Zero the output region
        for (i in 0 until headSize) {
            output[outputOffset + i] = 0f
        }

        for (t in 0..currentPos) {
            val cacheBase = layerBase + t * kvDim + headOffset
            val weight = weights[t]

            for (i in 0 until headSize) {
                output[outputOffset + i] += weight * valueFloatBuffer[cacheBase + i]
            }
        }
    }

    /**
     * Reset all cached values to zero.
     */
    override fun reset() {
        keyFloatBuffer.clear()
        valueFloatBuffer.clear()

        // Zero-fill the buffers
        val zeros = FloatArray(minOf(1024, floatsPerLayer))
        var remaining = nLayers * floatsPerLayer

        keyFloatBuffer.position(0)
        valueFloatBuffer.position(0)

        while (remaining > 0) {
            val chunk = minOf(zeros.size, remaining)
            keyFloatBuffer.put(zeros, 0, chunk)
            valueFloatBuffer.put(zeros, 0, chunk)
            remaining -= chunk
        }
    }

    /**
     * Get direct access to key buffer for a specific layer and position.
     * Returns the offset and allows direct float reads.
     */
    public fun getKeyOffset(layerIdx: Int, position: Int): Int {
        return (layerIdx * seqLen + position) * kvDim
    }

    /**
     * Get direct access to value buffer for a specific layer and position.
     */
    public fun getValueOffset(layerIdx: Int, position: Int): Int {
        return (layerIdx * seqLen + position) * kvDim
    }

    /**
     * Read a key vector at the specified position.
     */
    public fun getKeyVector(layerIdx: Int, position: Int, dest: FloatArray, destOffset: Int = 0) {
        val offset = (layerIdx * seqLen + position) * kvDim
        keyFloatBuffer.position(offset)
        keyFloatBuffer.get(dest, destOffset, kvDim)
    }

    /**
     * Read a value vector at the specified position.
     */
    public fun getValueVector(layerIdx: Int, position: Int, dest: FloatArray, destOffset: Int = 0) {
        val offset = (layerIdx * seqLen + position) * kvDim
        valueFloatBuffer.position(offset)
        valueFloatBuffer.get(dest, destOffset, kvDim)
    }

    /**
     * Get a key value at a specific index (KvCache interface).
     */
    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val index = (layerIdx * seqLen + position) * kvDim + headOffset + elementIdx
        return keyFloatBuffer[index]
    }

    /**
     * Get a value at a specific index (KvCache interface).
     */
    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val index = (layerIdx * seqLen + position) * kvDim + headOffset + elementIdx
        return valueFloatBuffer[index]
    }

    /**
     * Direct access to key float at given flat index.
     */
    public fun getKeyFloat(index: Int): Float = keyFloatBuffer[index]

    /**
     * Direct access to value float at given flat index.
     */
    public fun getValueFloat(index: Int): Float = valueFloatBuffer[index]

    /**
     * Release native memory.
     * After calling close(), the cache should not be used.
     */
    override fun close() {
        // Direct ByteBuffers are automatically deallocated when garbage collected,
        // but we can help by clearing references.
        // Note: There's no portable way to explicitly free direct buffer memory in Java.
        // The memory will be reclaimed when the buffers are GC'd.
        keyBuffer.clear()
        valueBuffer.clear()
    }

    /**
     * Memory statistics for monitoring.
     */
    public fun memoryStats(): MemoryStats = MemoryStats(
        nLayers = nLayers,
        seqLen = seqLen,
        kvDim = kvDim,
        totalBytes = totalBytes * 2  // keys + values
    )

    public data class MemoryStats(
        val nLayers: Int,
        val seqLen: Int,
        val kvDim: Int,
        val totalBytes: Long
    ) {
        val totalMegabytes: Double get() = totalBytes / (1024.0 * 1024.0)
    }
}
