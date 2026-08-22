package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Default KV cache implementation using dense FP32 storage.
 *
 * This is the reference/baseline implementation that stores K/V as
 * uncompressed float arrays. Quantized implementations (Q8_0, TurboQuant)
 * will override [appendToken] and [readKeys]/[readValues] with
 * encode-on-write / decode-on-read paths.
 *
 * Internal layout per layer:
 * - keys:   `FloatArray(numHeads * maxSeqLen * headDim)` — [numHeads, maxSeqLen, headDim]
 * - values: `FloatArray(numHeads * maxSeqLen * headDim)` — [numHeads, maxSeqLen, headDim]
 *
 * Append writes to position [currentSeqLen]; read returns a contiguous slice.
 */
public class DefaultKvCacheStore(
    private val config: KvCacheConfig
) : KvCacheStore {

    override val numLayers: Int get() = config.numLayers
    override val numHeads: Int get() = config.numHeads
    override val headDim: Int get() = config.headDim
    override val maxSeqLen: Int get() = config.maxSeqLen
    override val keyEncoding: TensorEncoding get() = config.keyEncoding
    override val valueEncoding: TensorEncoding get() = config.valueEncoding
    override val placement: Placement get() = config.placement

    private var _currentSeqLen: Int = 0
    override val currentSeqLen: Int get() = _currentSeqLen

    // Per-layer storage: keys[layer] and values[layer]
    // Each is [numHeads, maxSeqLen, headDim] laid out as contiguous float array
    private val keys: Array<FloatArray> = Array(numLayers) {
        FloatArray(numHeads * maxSeqLen * headDim)
    }
    private val values: Array<FloatArray> = Array(numLayers) {
        FloatArray(numHeads * maxSeqLen * headDim)
    }

    override fun appendToken(layer: Int, key: FloatArray, value: FloatArray) {
        requireLayerIndex(layer)
        check(_currentSeqLen < maxSeqLen) {
            "KV cache is full: currentSeqLen=$_currentSeqLen, maxSeqLen=$maxSeqLen"
        }
        require(key.size == numHeads * headDim) {
            "Key size mismatch: expected ${numHeads * headDim}, got ${key.size}"
        }
        require(value.size == numHeads * headDim) {
            "Value size mismatch: expected ${numHeads * headDim}, got ${value.size}"
        }

        val pos = _currentSeqLen
        val layerKeys = keys[layer]
        val layerValues = values[layer]

        // Copy each head's slice into the [head, pos, :] position
        for (h in 0 until numHeads) {
            val srcOffset = h * headDim
            val dstOffset = h * maxSeqLen * headDim + pos * headDim
            key.copyInto(layerKeys, dstOffset, srcOffset, srcOffset + headDim)
            value.copyInto(layerValues, dstOffset, srcOffset, srcOffset + headDim)
        }

        // Only increment seqLen when the last layer is written
        if (layer == numLayers - 1) {
            _currentSeqLen++
        }
    }

    override fun readKeys(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(keys[layer], layer, startPos, endPos)
    }

    override fun readValues(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(values[layer], layer, startPos, endPos)
    }

    override fun readKeyStorage(layer: Int, startPos: Int, endPos: Int): TensorStorage {
        return toTensorStorage(readKeys(layer, startPos, endPos), endPos - startPos, keyEncoding)
    }

    override fun readValueStorage(layer: Int, startPos: Int, endPos: Int): TensorStorage {
        return toTensorStorage(readValues(layer, startPos, endPos), endPos - startPos, valueEncoding)
    }

    override fun evict(fromPos: Int) {
        require(fromPos in 0..currentSeqLen) {
            "evict fromPos=$fromPos out of range [0, $currentSeqLen]"
        }
        _currentSeqLen = fromPos
        // Zero out evicted region for safety (prevents stale reads)
        for (layer in 0 until numLayers) {
            for (h in 0 until numHeads) {
                val offset = h * maxSeqLen * headDim + fromPos * headDim
                val count = (maxSeqLen - fromPos) * headDim
                keys[layer].fill(0f, offset, offset + count)
                values[layer].fill(0f, offset, offset + count)
            }
        }
    }

    override fun clear() {
        _currentSeqLen = 0
        for (layer in 0 until numLayers) {
            keys[layer].fill(0f)
            values[layer].fill(0f)
        }
    }

    override fun memoryReport(): KvCacheMemoryReport {
        val elementsPerLayer = numHeads.toLong() * maxSeqLen * headDim
        val logicalBytesPerLayer = elementsPerLayer * 4 // FP32
        return KvCacheMemoryReport(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            currentSeqLen = _currentSeqLen,
            keyEncoding = keyEncoding,
            valueEncoding = valueEncoding,
            placement = placement,
            keyPhysicalBytes = numLayers * logicalBytesPerLayer,
            valuePhysicalBytes = numLayers * logicalBytesPerLayer,
            keyLogicalBytes = numLayers * logicalBytesPerLayer,
            valueLogicalBytes = numLayers * logicalBytesPerLayer
        )
    }

    // --- Internal helpers ---

    private fun readRange(
        layerData: FloatArray,
        layer: Int,
        startPos: Int,
        endPos: Int
    ): FloatArray {
        requireLayerIndex(layer)
        require(startPos in 0..endPos) { "Invalid range: startPos=$startPos, endPos=$endPos" }
        require(endPos <= _currentSeqLen) {
            "endPos=$endPos exceeds currentSeqLen=$_currentSeqLen"
        }

        val seqLen = endPos - startPos
        val result = FloatArray(numHeads * seqLen * headDim)
        for (h in 0 until numHeads) {
            val srcBase = h * maxSeqLen * headDim + startPos * headDim
            val dstBase = h * seqLen * headDim
            layerData.copyInto(result, dstBase, srcBase, srcBase + seqLen * headDim)
        }
        return result
    }

    private fun toTensorStorage(
        data: FloatArray,
        seqLen: Int,
        encoding: TensorEncoding
    ): TensorStorage {
        // Convert FloatArray to ByteArray for TensorStorage
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val bits = data[i].toRawBits()
            bytes[i * 4] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return TensorStorage(
            shape = Shape(numHeads, seqLen, headDim),
            dtype = FP32,
            encoding = encoding,
            buffer = BufferHandle.Owned(bytes),
            placement = placement
        )
    }

    private fun requireLayerIndex(layer: Int) {
        require(layer in 0 until numLayers) {
            "Layer index $layer out of range [0, $numLayers)"
        }
    }
}
