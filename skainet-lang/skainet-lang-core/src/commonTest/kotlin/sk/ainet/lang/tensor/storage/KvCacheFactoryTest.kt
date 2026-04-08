package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.ops.turboquant.TurboQuantPresets
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Tests for KvCacheStore factory methods, TurboQuantPresets.forModel(),
 * and KvCacheAnnotationResolver.
 */
class KvCacheFactoryTest {

    // --- KvCacheStore.dense() ---

    @Test
    fun denseFactoryCreatesDenseStore() {
        val cache = KvCacheStore.dense(numLayers = 2, numHeads = 4, headDim = 64, maxSeqLen = 128)
        assertIs<DefaultKvCacheStore>(cache)
        assertEquals(2, cache.numLayers)
        assertEquals(4, cache.numHeads)
        assertEquals(64, cache.headDim)
        assertEquals(128, cache.maxSeqLen)
    }

    // --- KvCacheStore.turboQuant(preset) ---

    @Test
    fun turboQuantPresetBalanced() {
        val cache = KvCacheStore.turboQuant("balanced", 2, 4, 64, 128)
        assertIs<TurboQuantKvCacheStore>(cache)
        assertEquals(2, cache.numLayers)
        assertIs<TensorEncoding.TurboQuantPolar>(cache.keyEncoding)
        assertIs<TensorEncoding.TurboQuantPolar>(cache.valueEncoding)
        assertEquals(4, (cache.keyEncoding as TensorEncoding.TurboQuantPolar).bitsPerElement)
    }

    @Test
    fun turboQuantPresetSafeLowbit() {
        val cache = KvCacheStore.turboQuant("safe-lowbit", 2, 4, 64, 128)
        assertIs<TurboQuantKvCacheStore>(cache)
        assertEquals(TensorEncoding.Q8_0, cache.keyEncoding)
        assertIs<TensorEncoding.TurboQuantPolar>(cache.valueEncoding)
    }

    @Test
    fun turboQuantPresetExperimentalMax() {
        val cache = KvCacheStore.turboQuant("experimental-max", 2, 4, 64, 128)
        assertIs<TurboQuantKvCacheStore>(cache)
        assertEquals(3, (cache.keyEncoding as TensorEncoding.TurboQuantPolar).bitsPerElement)
    }

    @Test
    fun turboQuantUnknownPresetThrows() {
        assertFailsWith<IllegalArgumentException> {
            KvCacheStore.turboQuant("nonexistent", 2, 4, 64, 128)
        }
    }

    // --- KvCacheStore.turboQuant(custom) ---

    @Test
    fun turboQuantCustomBits() {
        val cache = KvCacheStore.turboQuant(
            numLayers = 2, numHeads = 4, headDim = 64, maxSeqLen = 128,
            keyBits = 8, valueBits = 3
        )
        assertIs<TurboQuantKvCacheStore>(cache)
        assertEquals(8, (cache.keyEncoding as TensorEncoding.TurboQuantPolar).bitsPerElement)
        assertEquals(3, (cache.valueEncoding as TensorEncoding.TurboQuantPolar).bitsPerElement)
    }

    // --- KvCacheStore.fromPreset() ---

    @Test
    fun fromPresetCreatesCorrectCache() {
        val preset = TurboQuantPresets.balanced(2, 4, 64, 128)
        val cache = KvCacheStore.fromPreset(preset)
        assertIs<TurboQuantKvCacheStore>(cache)
        assertEquals(2, cache.numLayers)
    }

    // --- TurboQuantPresets.forModel() ---

    @Test
    fun forModelBalanced() {
        val preset = TurboQuantPresets.forModel("balanced", 32, 32, 128, 4096)
        assertEquals("balanced", preset.name)
        assertEquals(32, preset.cacheConfig.numLayers)
        assertEquals(4096, preset.cacheConfig.maxSeqLen)
    }

    @Test
    fun forModelUnknownThrows() {
        assertFailsWith<IllegalArgumentException> {
            TurboQuantPresets.forModel("invalid", 2, 4, 64, 128)
        }
    }

    // --- KvCacheAnnotationResolver ---

    @Test
    fun resolvePresetString() {
        val cache = KvCacheAnnotationResolver.resolve("balanced", 2, 4, 64, 128)
        assertIs<TurboQuantKvCacheStore>(cache)
    }

    @Test
    fun resolveDensePreset() {
        val cache = KvCacheAnnotationResolver.resolve("dense", 2, 4, 64, 128)
        assertIs<DefaultKvCacheStore>(cache)
    }

    @Test
    fun resolveNonePreset() {
        val cache = KvCacheAnnotationResolver.resolve("none", 2, 4, 64, 128)
        assertIs<DefaultKvCacheStore>(cache)
    }

    // --- End-to-end: factory → append → read ---

    @Test
    fun factoryCreatedCacheWorksEndToEnd() {
        val cache = KvCacheStore.turboQuant("balanced", 1, 2, 64, 16)
        val bridge = CompressedKvAttention(cache)

        val key = FloatArray(2 * 64) { it.toFloat() / 128f }
        val value = FloatArray(2 * 64) { -it.toFloat() / 128f }

        bridge.storeKeyValue(0, key, value)
        assertEquals(1, cache.currentSeqLen)

        val readK = bridge.loadKeysForAttention(0)
        assertEquals(2 * 1 * 64, readK.size)

        val report = cache.memoryReport()
        assertTrue(report.compressionRatio > 1.0,
            "TurboQuant should compress: ratio=${report.compressionRatio}")
    }
}
