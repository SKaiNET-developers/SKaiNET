@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

package sk.ainet.lang.tensor.storage

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for [CompressedKvAttention] — the bridge between KvCacheStore and SDPA.
 */
class CompressedKvAttentionTest {

    private fun createBridge(
        numLayers: Int = 1,
        numHeads: Int = 2,
        headDim: Int = 4,
        maxSeqLen: Int = 8,
        keyEncoding: TensorEncoding = TensorEncoding.Dense(4),
        valueEncoding: TensorEncoding = TensorEncoding.Dense(4),
        strategy: CompressedKvAttention.DequantStrategy = CompressedKvAttention.DequantStrategy.FULL_TILE
    ): CompressedKvAttention {
        val config = KvCacheConfig(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            keyEncoding = keyEncoding,
            valueEncoding = valueEncoding
        )
        return CompressedKvAttention(DefaultKvCacheStore(config), strategy)
    }

    @Test
    fun storeAndLoadRoundTrip() {
        val bridge = createBridge()
        val key = FloatArray(2 * 4) { it.toFloat() }
        val value = FloatArray(2 * 4) { (it + 100).toFloat() }

        bridge.storeKeyValue(0, key, value)

        val loadedKeys = bridge.loadKeysForAttention(0)
        val loadedValues = bridge.loadValuesForAttention(0)

        assertEquals(2 * 1 * 4, loadedKeys.size)
        assertEquals(0f, loadedKeys[0])
        assertEquals(7f, loadedKeys[7])

        assertEquals(100f, loadedValues[0])
        assertEquals(107f, loadedValues[7])
    }

    @Test
    fun loadWithSubRange() {
        val bridge = createBridge(numHeads = 1, headDim = 2)

        bridge.storeKeyValue(0, floatArrayOf(1f, 2f), floatArrayOf(10f, 20f))
        bridge.storeKeyValue(0, floatArrayOf(3f, 4f), floatArrayOf(30f, 40f))
        bridge.storeKeyValue(0, floatArrayOf(5f, 6f), floatArrayOf(50f, 60f))

        // Read only position 1
        val keys = bridge.loadKeysForAttention(0, startPos = 1, endPos = 2)
        assertEquals(2, keys.size)
        assertEquals(3f, keys[0])
        assertEquals(4f, keys[1])
    }

    @Test
    fun rawStorageReturnsTensorStorage() {
        val bridge = createBridge()
        bridge.storeKeyValue(0, FloatArray(8), FloatArray(8))

        val keyStorage = bridge.loadKeyStorageRaw(0)
        assertEquals(LogicalDType.FLOAT32, keyStorage.logicalType)
        assertEquals(Ownership.OWNED, keyStorage.ownership)

        val valueStorage = bridge.loadValueStorageRaw(0)
        assertEquals(LogicalDType.FLOAT32, valueStorage.logicalType)
    }

    @Test
    fun isCompressedDetectsEncoding() {
        val denseBridge = createBridge()
        assertFalse(denseBridge.isKeyCompressed)
        assertFalse(denseBridge.isValueCompressed)

        val compressedBridge = createBridge(
            keyEncoding = TensorEncoding.Q8_0,
            valueEncoding = TensorEncoding.Q4_K
        )
        assertTrue(compressedBridge.isKeyCompressed)
        assertTrue(compressedBridge.isValueCompressed)
    }

    @Test
    fun asymmetricCompression() {
        val bridge = createBridge(
            keyEncoding = TensorEncoding.Q8_0,
            valueEncoding = TensorEncoding.Dense(4)
        )
        assertTrue(bridge.isKeyCompressed)
        assertFalse(bridge.isValueCompressed)
    }

    @Test
    fun rawStorageStrategyFallsBackToFloat() {
        val bridge = createBridge(
            strategy = CompressedKvAttention.DequantStrategy.RAW_STORAGE
        )
        bridge.storeKeyValue(0, FloatArray(8) { it.toFloat() }, FloatArray(8))

        // RAW_STORAGE still returns float for default implementation
        val keys = bridge.loadKeysForAttention(0)
        assertEquals(8, keys.size)
        assertEquals(0f, keys[0])
    }
}
