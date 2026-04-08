package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for quantization shape and size utilities in [Quants.kt].
 */
class QuantsTest {

    // --- quantShapeToByteShape ---

    @Test
    fun quantShapeToByteShape_Q4_K() {
        val shape = listOf(32UL, 256UL)
        val result = quantShapeToByteShape(shape, GGMLQuantizationType.Q4_K)
        // Q4_K: blockSize=256, typeSize=144 → 256/256 * 144 = 144
        assertEquals(listOf(32UL, 144UL), result)
    }

    @Test
    fun quantShapeToByteShape_Q8_0() {
        val shape = listOf(128UL)
        val result = quantShapeToByteShape(shape, GGMLQuantizationType.Q8_0)
        // Q8_0: blockSize=32, typeSize=34 → 128/32 * 34 = 136
        assertEquals(listOf(136UL), result)
    }

    @Test
    fun quantShapeToByteShape_F32_passthrough() {
        val shape = listOf(10UL, 20UL)
        val result = quantShapeToByteShape(shape, GGMLQuantizationType.F32)
        // F32: blockSize=1, typeSize=4 → 20/1 * 4 = 80
        assertEquals(listOf(10UL, 80UL), result)
    }

    @Test
    fun quantShapeToByteShape_unaligned_throws() {
        assertFailsWith<IllegalArgumentException> {
            quantShapeToByteShape(listOf(100UL), GGMLQuantizationType.Q4_K) // 100 not multiple of 256
        }
    }

    // --- byteShapeToQuantShape ---

    @Test
    fun byteShapeToQuantShape_Q4_K() {
        val byteShape = listOf(32UL, 144UL)
        val result = byteShapeToQuantShape(byteShape, GGMLQuantizationType.Q4_K)
        assertEquals(listOf(32UL, 256UL), result)
    }

    @Test
    fun byteShapeToQuantShape_Q8_0() {
        val byteShape = listOf(136UL)
        val result = byteShapeToQuantShape(byteShape, GGMLQuantizationType.Q8_0)
        assertEquals(listOf(128UL), result)
    }

    @Test
    fun byteShapeToQuantShape_roundTrip() {
        val original = listOf(16UL, 512UL)
        val byteShape = quantShapeToByteShape(original, GGMLQuantizationType.Q4_K)
        val recovered = byteShapeToQuantShape(byteShape, GGMLQuantizationType.Q4_K)
        assertEquals(original, recovered)
    }

    @Test
    fun byteShapeToQuantShape_unaligned_throws() {
        assertFailsWith<IllegalArgumentException> {
            byteShapeToQuantShape(listOf(100UL), GGMLQuantizationType.Q8_0) // 100 not multiple of 34
        }
    }

    // --- quantElementCount ---

    @Test
    fun quantElementCount_standard() {
        assertEquals(1024UL, quantElementCount(listOf(32UL, 32UL)))
    }

    @Test
    fun quantElementCount_scalar() {
        assertEquals(1UL, quantElementCount(emptyList()))
    }

    @Test
    fun quantElementCount_1d() {
        assertEquals(256UL, quantElementCount(listOf(256UL)))
    }

    // --- quantByteSize ---

    @Test
    fun quantByteSize_Q4_K() {
        // 256 elements → 1 block → 144 bytes
        assertEquals(144UL, quantByteSize(256UL, GGMLQuantizationType.Q4_K))
    }

    @Test
    fun quantByteSize_Q8_0() {
        // 64 elements → 2 blocks → 68 bytes
        assertEquals(68UL, quantByteSize(64UL, GGMLQuantizationType.Q8_0))
    }

    @Test
    fun quantByteSize_F32() {
        assertEquals(40UL, quantByteSize(10UL, GGMLQuantizationType.F32))
    }

    // --- isBlockQuantized ---

    @Test
    fun isBlockQuantized_true() {
        assertTrue(isBlockQuantized(GGMLQuantizationType.Q4_K))
        assertTrue(isBlockQuantized(GGMLQuantizationType.Q8_0))
        assertTrue(isBlockQuantized(GGMLQuantizationType.Q2_K))
        assertTrue(isBlockQuantized(GGMLQuantizationType.TQ2_0))
    }

    @Test
    fun isBlockQuantized_false() {
        assertFalse(isBlockQuantized(GGMLQuantizationType.F32))
        assertFalse(isBlockQuantized(GGMLQuantizationType.F16))
        assertFalse(isBlockQuantized(GGMLQuantizationType.I8))
    }

    // --- quantBlockSize / quantTypeSize ---

    @Test
    fun quantBlockSize_known() {
        assertEquals(256, quantBlockSize(GGMLQuantizationType.Q4_K))
        assertEquals(32, quantBlockSize(GGMLQuantizationType.Q8_0))
        assertEquals(1, quantBlockSize(GGMLQuantizationType.F32))
    }

    @Test
    fun quantTypeSize_known() {
        assertEquals(144, quantTypeSize(GGMLQuantizationType.Q4_K))
        assertEquals(34, quantTypeSize(GGMLQuantizationType.Q8_0))
        assertEquals(4, quantTypeSize(GGMLQuantizationType.F32))
    }

    @Test
    fun quantBlockSize_unknown_returns_null() {
        assertEquals(null, quantBlockSize(GGMLQuantizationType.UNKNOWN))
    }

    // --- validateQuantizedBytes ---

    @Test
    fun validateQuantizedBytes_correct_size() {
        val bytes = ByteArray(144) // 1 Q4_K block
        validateQuantizedBytes(bytes, 256UL, GGMLQuantizationType.Q4_K)
    }

    @Test
    fun validateQuantizedBytes_wrong_size_throws() {
        assertFailsWith<IllegalArgumentException> {
            validateQuantizedBytes(ByteArray(100), 256UL, GGMLQuantizationType.Q4_K)
        }
    }

    // --- Coverage for all quant types in GGML_QUANT_SIZES ---

    @Test
    fun allQuantSizesHaveBlockAndTypeSize() {
        for ((type, sizes) in GGML_QUANT_SIZES) {
            val (blockSize, typeSize) = sizes
            assertTrue(blockSize > 0, "Block size for $type must be positive")
            assertTrue(typeSize > 0, "Type size for $type must be positive")
            assertNotNull(quantBlockSize(type))
            assertNotNull(quantTypeSize(type))
        }
    }
}
