package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.ops.turboquant.TurboQuantConfig
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for [TurboQuantKvCacheStore] — the compressed KV cache.
 */
class TurboQuantKvCacheStoreTest {

    private fun createStore(
        numLayers: Int = 1,
        numHeads: Int = 2,
        headDim: Int = 64,
        maxSeqLen: Int = 16,
        bits: Int = 4,
        useQjl: Boolean = false
    ): TurboQuantKvCacheStore {
        val config = KvCacheConfig(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = bits),
            valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = bits)
        )
        val quantConfig = if (useQjl) {
            TurboQuantConfig.polarPlusQjl(bits = bits)
        } else {
            TurboQuantConfig.polarOnly(bits = bits)
        }
        return TurboQuantKvCacheStore(config, quantConfig, quantConfig)
    }

    @Test
    fun appendAndReadBasic() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64)
        val key = FloatArray(64) { (it - 32).toFloat() / 32f }
        val value = FloatArray(64) { (it - 32).toFloat() / 64f }

        store.appendToken(0, key, value)
        assertEquals(1, store.currentSeqLen)

        val readK = store.readKeys(0)
        val readV = store.readValues(0)

        assertEquals(64, readK.size)
        assertEquals(64, readV.size)

        // Check reconstruction accuracy (4-bit should be reasonable)
        var maxKeyError = 0f
        for (i in key.indices) {
            maxKeyError = maxOf(maxKeyError, abs(key[i] - readK[i]))
        }
        assertTrue(maxKeyError < 0.5f,
            "4-bit TurboQuant key reconstruction error should be < 0.5, got $maxKeyError")
    }

    @Test
    fun multipleTokens() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64, maxSeqLen = 8)

        for (t in 0 until 4) {
            val key = FloatArray(64) { (it + t).toFloat() / 64f }
            val value = FloatArray(64) { (it - t).toFloat() / 64f }
            store.appendToken(0, key, value)
        }

        assertEquals(4, store.currentSeqLen)

        val allKeys = store.readKeys(0)
        // [numHeads=1, seqLen=4, headDim=64]
        assertEquals(1 * 4 * 64, allKeys.size)
    }

    @Test
    fun multipleHeads() {
        val store = createStore(numLayers = 1, numHeads = 4, headDim = 64)

        val key = FloatArray(4 * 64) { it.toFloat() / 256f }
        val value = FloatArray(4 * 64) { -it.toFloat() / 256f }
        store.appendToken(0, key, value)

        val readK = store.readKeys(0)
        assertEquals(4 * 1 * 64, readK.size)
    }

    @Test
    fun rangRead() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64, maxSeqLen = 8)

        for (t in 0 until 4) {
            store.appendToken(0, FloatArray(64) { t.toFloat() }, FloatArray(64))
        }

        val partial = store.readKeys(0, startPos = 1, endPos = 3)
        assertEquals(1 * 2 * 64, partial.size) // 2 positions
    }

    @Test
    fun eviction() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64, maxSeqLen = 8)

        for (t in 0 until 4) {
            store.appendToken(0, FloatArray(64), FloatArray(64))
        }
        assertEquals(4, store.currentSeqLen)

        store.evict(2)
        assertEquals(2, store.currentSeqLen)
    }

    @Test
    fun clear() {
        val store = createStore()
        store.appendToken(0, FloatArray(2 * 64), FloatArray(2 * 64))
        assertEquals(1, store.currentSeqLen)

        store.clear()
        assertEquals(0, store.currentSeqLen)
    }

    @Test
    fun capacityOverflow() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64, maxSeqLen = 2)
        store.appendToken(0, FloatArray(64), FloatArray(64))
        store.appendToken(0, FloatArray(64), FloatArray(64))

        assertFailsWith<IllegalStateException> {
            store.appendToken(0, FloatArray(64), FloatArray(64))
        }
    }

    @Test
    fun compressionRatio() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 128, maxSeqLen = 8, bits = 4)

        for (t in 0 until 4) {
            store.appendToken(0, FloatArray(128) { it.toFloat() }, FloatArray(128))
        }

        val report = store.memoryReport()
        // 4-bit should compress significantly vs FP32
        assertTrue(report.compressionRatio > 1.5,
            "4-bit TurboQuant should compress at least 1.5x, got ${report.compressionRatio}")
    }

    @Test
    fun qjlVariant() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 64, bits = 4, useQjl = true)

        val key = FloatArray(64) { (it - 32).toFloat() / 32f }
        store.appendToken(0, key, FloatArray(64))

        val readK = store.readKeys(0)
        assertEquals(64, readK.size)
    }

    @Test
    fun multipleLayers() {
        val store = createStore(numLayers = 2, numHeads = 1, headDim = 64)

        val key0 = FloatArray(64) { 1f }
        val key1 = FloatArray(64) { -1f }

        store.appendToken(0, key0, FloatArray(64))
        store.appendToken(1, key1, FloatArray(64))

        assertEquals(1, store.currentSeqLen)

        val readK0 = store.readKeys(0)
        val readK1 = store.readKeys(1)

        // Layer 0 should reconstruct toward positive, layer 1 toward negative
        val avgK0 = readK0.sum() / readK0.size
        val avgK1 = readK1.sum() / readK1.size
        assertTrue(avgK0 > avgK1, "Layer 0 (pos) avg ($avgK0) should > layer 1 (neg) avg ($avgK1)")
    }

    @Test
    fun memoryReportAccurate() {
        val store = createStore(numLayers = 2, numHeads = 2, headDim = 64, maxSeqLen = 8)
        store.appendToken(0, FloatArray(128), FloatArray(128))
        store.appendToken(1, FloatArray(128), FloatArray(128))

        val report = store.memoryReport()
        assertEquals(2, report.numLayers)
        assertEquals(2, report.numHeads)
        assertEquals(64, report.headDim)
        assertEquals(1, report.currentSeqLen)
        assertTrue(report.totalPhysicalBytes > 0)
        assertTrue(report.totalLogicalBytes > 0)
    }

    @Test
    fun asymmetricKeyValueConfig() {
        val config = KvCacheConfig(
            numLayers = 1, numHeads = 1, headDim = 64, maxSeqLen = 8,
            keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 8),
            valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4)
        )
        val store = TurboQuantKvCacheStore(
            config,
            keyConfig = TurboQuantConfig.polarOnly(bits = 8),
            valueConfig = TurboQuantConfig.polarOnly(bits = 4)
        )

        val input = FloatArray(64) { (it - 32).toFloat() / 32f }
        store.appendToken(0, input, input)

        val readK = store.readKeys(0)
        val readV = store.readValues(0)

        // 8-bit keys should be more accurate than 4-bit values
        var keyError = 0f
        var valError = 0f
        for (i in input.indices) {
            keyError += abs(input[i] - readK[i])
            valError += abs(input[i] - readV[i])
        }
        assertTrue(keyError < valError,
            "8-bit keys ($keyError) should have less error than 4-bit values ($valError)")
    }
}
