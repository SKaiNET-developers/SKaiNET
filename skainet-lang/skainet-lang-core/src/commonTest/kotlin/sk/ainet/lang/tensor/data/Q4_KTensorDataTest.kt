package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals
import kotlin.math.abs

class Q4_KTensorDataTest {

    /**
     * Helper to create a Q4_K block with specified parameters.
     *
     * Block layout (144 bytes):
     * - [0..1]: f16 d (main scale)
     * - [2..3]: f16 dMin (min scale)
     * - [4..15]: 12 bytes packed scale/min indices
     * - [16..143]: 128 bytes 4-bit codes
     */
    private fun createQ4KBlock(
        d: Short,
        dMin: Short,
        scaleMinIndices: ByteArray = ByteArray(12),
        codes: ByteArray = ByteArray(128)
    ): ByteArray {
        val block = ByteArray(144)

        // d (little-endian)
        block[0] = (d.toInt() and 0xFF).toByte()
        block[1] = ((d.toInt() shr 8) and 0xFF).toByte()

        // dMin (little-endian)
        block[2] = (dMin.toInt() and 0xFF).toByte()
        block[3] = ((dMin.toInt() shr 8) and 0xFF).toByte()

        // Scale/min indices
        scaleMinIndices.copyInto(block, 4, 0, minOf(12, scaleMinIndices.size))

        // Codes
        codes.copyInto(block, 16, 0, minOf(128, codes.size))

        return block
    }

    @Test
    fun `Q4_KBlockTensorData can read d and dMin from block`() {
        // d = 1.0 (f16 0x3C00), dMin = 0.5 (f16 0x3800)
        val block = createQ4KBlock(0x3C00, 0x3800)

        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        assertEquals(1.0f, tensor.getBlockD(0), 0.001f)
        assertEquals(0.5f, tensor.getBlockDMin(0), 0.001f)
    }

    @Test
    fun `Q4_KBlockTensorData can read 4-bit codes`() {
        val codes = ByteArray(128)
        // First byte: low nibble = 5, high nibble = 10
        codes[0] = ((10 shl 4) or 5).toByte()  // 0xA5
        // Second byte: low nibble = 3, high nibble = 12
        codes[1] = ((12 shl 4) or 3).toByte()  // 0xC3

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        assertEquals(5, tensor.getCode(0, 0))
        assertEquals(10, tensor.getCode(0, 1))
        assertEquals(3, tensor.getCode(0, 2))
        assertEquals(12, tensor.getCode(0, 3))
    }

    @Test
    fun `Q4_KBlockTensorData handles all 4-bit values`() {
        val codes = ByteArray(128)
        // Pack all values 0-15 twice each
        for (i in 0 until 16) {
            codes[i] = ((i shl 4) or i).toByte()
        }

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        for (i in 0 until 16) {
            assertEquals(i, tensor.getCode(0, i * 2), "Code at even index ${i * 2}")
            assertEquals(i, tensor.getCode(0, i * 2 + 1), "Code at odd index ${i * 2 + 1}")
        }
    }

    @Test
    fun `Q4_KBlockTensorData get via indices works`() {
        val codes = ByteArray(128)
        codes[0] = 0x21  // element 0 = 1, element 1 = 2

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        assertEquals(1.toByte(), tensor[0])
        assertEquals(2.toByte(), tensor[1])
    }

    @Test
    fun `Q4_KBlockTensorData 2D access works correctly`() {
        val codes = ByteArray(128)
        // Fill with sequential values mod 16
        for (i in 0 until 128) {
            val lo = (i * 2) % 16
            val hi = (i * 2 + 1) % 16
            codes[i] = ((hi shl 4) or lo).toByte()
        }

        // 16x16 = 256 elements = 1 block
        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(16, 16), block)

        // tensor[0, 0] = element 0 = 0
        // tensor[0, 1] = element 1 = 1
        // tensor[1, 0] = element 16 = 0 (mod 16)
        assertEquals(0, tensor[0, 0].toInt())
        assertEquals(1, tensor[0, 1].toInt())
    }

    @Test
    fun `Q4_KBlockTensorData set operation works`() {
        val block = createQ4KBlock(0x3C00, 0x0000)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        tensor[0] = 7
        tensor[1] = 11

        assertEquals(7.toByte(), tensor[0])
        assertEquals(11.toByte(), tensor[1])
    }

    @Test
    fun `Q4_KBlockTensorData handles multiple blocks`() {
        // 512 elements = 2 blocks
        val block1 = createQ4KBlock(0x3C00, 0x3800)  // d=1.0, dMin=0.5
        val block2 = createQ4KBlock(0x4000, 0x3C00)  // d=2.0, dMin=1.0

        val blockData = block1 + block2
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(512), blockData)

        assertEquals(2, tensor.blockCount)
        assertEquals(1.0f, tensor.getBlockD(0), 0.001f)
        assertEquals(0.5f, tensor.getBlockDMin(0), 0.001f)
        assertEquals(2.0f, tensor.getBlockD(1), 0.001f)
        assertEquals(1.0f, tensor.getBlockDMin(1), 0.001f)
    }

    @Test
    fun `Q4_KBlockTensorData shape is preserved`() {
        val block = createQ4KBlock(0x3C00, 0x0000)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(16, 16), block)

        assertContentEquals(intArrayOf(16, 16), tensor.shape.dimensions)
        assertEquals(256, tensor.shape.volume)
        assertEquals(1, tensor.blockCount)
    }

    @Test
    fun `Q4_KTensorData constants are correct`() {
        assertEquals(256, Q4_KTensorData.BLOCK_SIZE)
        assertEquals(32, Q4_KTensorData.SUB_BLOCK_SIZE)
        assertEquals(8, Q4_KTensorData.SUB_BLOCKS_PER_BLOCK)
        assertEquals(144, Q4_KTensorData.BYTES_PER_BLOCK)
    }

    @Test
    fun `Q4_KBlockTensorData toFloatArray produces expected values`() {
        // Create a simple block where we can verify output
        // d = 1.0, dMin = 0.0, all scale/min indices = 63 (max)
        // This gives scale = 1.0 * (63/63) = 1.0, min = 0
        val scaleMinIndices = ByteArray(12) { 0xFF.toByte() }  // All 1s gives max indices

        val codes = ByteArray(128)
        codes[0] = 0x21  // element 0 = 1, element 1 = 2

        val block = createQ4KBlock(0x3C00, 0x0000, scaleMinIndices, codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        val floats = tensor.toFloatArray()

        // With scale = d * (63/63) = 1.0 and min = 0:
        // output[i] = code[i] * 1.0 + 0 = code[i]
        assertEquals(1.0f, floats[0], 0.1f)
        assertEquals(2.0f, floats[1], 0.1f)
    }

    @Test
    fun `Q4_KBlockTensorData sub-block boundaries are handled`() {
        val block = createQ4KBlock(0x3C00, 0x0000)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        // Sub-block 0: elements 0-31
        // Sub-block 1: elements 32-63
        // ...
        // Sub-block 7: elements 224-255

        // Verify we can access elements at sub-block boundaries
        for (subBlock in 0 until 8) {
            val startElem = subBlock * 32
            val endElem = startElem + 31

            // Access should not throw
            tensor.getCode(0, startElem)
            tensor.getCode(0, endElem)
        }
    }
}
