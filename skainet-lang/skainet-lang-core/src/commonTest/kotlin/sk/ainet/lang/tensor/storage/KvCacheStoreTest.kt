package sk.ainet.lang.tensor.storage

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for [KvCacheStore] contract and [DefaultKvCacheStore] implementation.
 */
class KvCacheStoreTest {

    private fun createStore(
        numLayers: Int = 2,
        numHeads: Int = 4,
        headDim: Int = 8,
        maxSeqLen: Int = 16
    ): DefaultKvCacheStore = DefaultKvCacheStore(
        KvCacheConfig(numLayers, numHeads, headDim, maxSeqLen)
    )

    // --- Append and read ---

    @Test
    fun appendAndReadSingleToken() {
        val store = createStore(numLayers = 1, numHeads = 2, headDim = 4, maxSeqLen = 8)
        val key = FloatArray(2 * 4) { it.toFloat() }   // [0..7]
        val value = FloatArray(2 * 4) { (it + 10).toFloat() } // [10..17]

        store.appendToken(0, key, value)
        assertEquals(1, store.currentSeqLen)

        val readK = store.readKeys(0)
        val readV = store.readValues(0)

        // Shape: [numHeads=2, seqLen=1, headDim=4]
        assertEquals(2 * 1 * 4, readK.size)
        assertEquals(2 * 1 * 4, readV.size)

        // Head 0: [0, 1, 2, 3]
        assertEquals(0f, readK[0])
        assertEquals(1f, readK[1])
        assertEquals(2f, readK[2])
        assertEquals(3f, readK[3])

        // Head 1: [4, 5, 6, 7]
        assertEquals(4f, readK[4])
        assertEquals(5f, readK[5])

        // Values: head 0 starts at 10
        assertEquals(10f, readV[0])
        assertEquals(13f, readV[3])
    }

    @Test
    fun appendMultipleTokens() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 2, maxSeqLen = 4)

        // Token 0
        store.appendToken(0, floatArrayOf(1f, 2f), floatArrayOf(10f, 20f))
        // Token 1
        store.appendToken(0, floatArrayOf(3f, 4f), floatArrayOf(30f, 40f))

        assertEquals(2, store.currentSeqLen)

        val keys = store.readKeys(0)
        // [numHeads=1, seqLen=2, headDim=2] = [1, 2, 3, 4]
        assertEquals(4, keys.size)
        assertEquals(1f, keys[0])
        assertEquals(2f, keys[1])
        assertEquals(3f, keys[2])
        assertEquals(4f, keys[3])
    }

    @Test
    fun appendMultipleLayers() {
        val store = createStore(numLayers = 2, numHeads = 1, headDim = 2, maxSeqLen = 4)

        // Layer 0 then Layer 1 for token 0
        store.appendToken(0, floatArrayOf(1f, 2f), floatArrayOf(10f, 20f))
        store.appendToken(1, floatArrayOf(5f, 6f), floatArrayOf(50f, 60f))

        assertEquals(1, store.currentSeqLen)

        // Layer 0 keys
        val k0 = store.readKeys(0)
        assertEquals(1f, k0[0])
        assertEquals(2f, k0[1])

        // Layer 1 keys
        val k1 = store.readKeys(1)
        assertEquals(5f, k1[0])
        assertEquals(6f, k1[1])
    }

    // --- Range reads ---

    @Test
    fun readSubRange() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 2, maxSeqLen = 8)

        // Append 4 tokens
        for (i in 0 until 4) {
            store.appendToken(0, floatArrayOf(i.toFloat(), (i + 10).toFloat()), floatArrayOf(0f, 0f))
        }

        // Read only positions 1..3
        val keys = store.readKeys(0, startPos = 1, endPos = 3)
        assertEquals(4, keys.size) // [1, numHeads=1] * 2 positions * headDim=2
        assertEquals(1f, keys[0])
        assertEquals(11f, keys[1])
        assertEquals(2f, keys[2])
        assertEquals(12f, keys[3])
    }

    // --- TensorStorage output ---

    @Test
    fun readKeyStorageReturnsTensorStorage() {
        val store = createStore(numLayers = 1, numHeads = 2, headDim = 4, maxSeqLen = 8)
        store.appendToken(0, FloatArray(8) { it.toFloat() }, FloatArray(8))

        val storage = store.readKeyStorage(0)
        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(Ownership.OWNED, storage.ownership)
        assertTrue(storage.isMutable)
    }

    // --- Eviction ---

    @Test
    fun evictTruncatesCache() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 2, maxSeqLen = 8)

        for (i in 0 until 4) {
            store.appendToken(0, floatArrayOf(i.toFloat(), 0f), floatArrayOf(0f, 0f))
        }
        assertEquals(4, store.currentSeqLen)

        store.evict(fromPos = 2)
        assertEquals(2, store.currentSeqLen)

        val keys = store.readKeys(0)
        assertEquals(4, keys.size) // 2 positions * headDim=2
        assertEquals(0f, keys[0])
        assertEquals(1f, keys[2])
    }

    @Test
    fun clearResetsEverything() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 2, maxSeqLen = 4)
        store.appendToken(0, floatArrayOf(1f, 2f), floatArrayOf(3f, 4f))
        assertEquals(1, store.currentSeqLen)

        store.clear()
        assertEquals(0, store.currentSeqLen)
    }

    // --- Capacity ---

    @Test
    fun appendBeyondCapacityThrows() {
        val store = createStore(numLayers = 1, numHeads = 1, headDim = 2, maxSeqLen = 2)
        store.appendToken(0, floatArrayOf(1f, 2f), floatArrayOf(3f, 4f))
        store.appendToken(0, floatArrayOf(5f, 6f), floatArrayOf(7f, 8f))

        assertFailsWith<IllegalStateException> {
            store.appendToken(0, floatArrayOf(9f, 10f), floatArrayOf(11f, 12f))
        }
    }

    // --- Validation ---

    @Test
    fun invalidLayerIndexThrows() {
        val store = createStore(numLayers = 2)
        assertFailsWith<IllegalArgumentException> {
            store.appendToken(5, FloatArray(store.numHeads * store.headDim), FloatArray(store.numHeads * store.headDim))
        }
    }

    @Test
    fun wrongKeySizeThrows() {
        val store = createStore(numLayers = 1, numHeads = 2, headDim = 4)
        assertFailsWith<IllegalArgumentException> {
            store.appendToken(0, FloatArray(3), FloatArray(8)) // wrong key size
        }
    }

    // --- Memory report ---

    @Test
    fun memoryReportIsAccurate() {
        val store = createStore(numLayers = 2, numHeads = 4, headDim = 8, maxSeqLen = 16)
        store.appendToken(0, FloatArray(32), FloatArray(32))
        store.appendToken(1, FloatArray(32), FloatArray(32))

        val report = store.memoryReport()
        assertEquals(2, report.numLayers)
        assertEquals(4, report.numHeads)
        assertEquals(8, report.headDim)
        assertEquals(16, report.maxSeqLen)
        assertEquals(1, report.currentSeqLen)
        assertEquals(TensorEncoding.Dense(4), report.keyEncoding)
        assertTrue(report.totalPhysicalBytes > 0)
        assertTrue(report.utilizationRatio > 0.0)
        assertTrue(report.utilizationRatio < 1.0)
    }

    // --- Config validation ---

    @Test
    fun invalidConfigThrows() {
        assertFailsWith<IllegalArgumentException> {
            KvCacheConfig(numLayers = 0, numHeads = 4, headDim = 8, maxSeqLen = 16)
        }
        assertFailsWith<IllegalArgumentException> {
            KvCacheConfig(numLayers = 2, numHeads = 0, headDim = 8, maxSeqLen = 16)
        }
    }

    // --- Asymmetric K/V encoding config ---

    @Test
    fun asymmetricConfigPreservesEncodings() {
        val config = KvCacheConfig(
            numLayers = 2,
            numHeads = 4,
            headDim = 64,
            maxSeqLen = 512,
            keyEncoding = TensorEncoding.Q8_0,
            valueEncoding = TensorEncoding.Q4_K
        )
        val store = DefaultKvCacheStore(config)
        assertEquals(TensorEncoding.Q8_0, store.keyEncoding)
        assertEquals(TensorEncoding.Q4_K, store.valueEncoding)
    }
}
