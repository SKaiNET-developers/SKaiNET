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
    fun `Q4_KBlockTensorData can read 4-bit codes (canonical strided layout)`() {
        // ggml strided layout: byte at offset (i within a 32-byte qs group)
        // holds element i in lo nibble, element i+32 in hi nibble.
        val codes = ByteArray(128)
        // First byte: low nibble = 5 (element 0), high nibble = 10 (element 32)
        codes[0] = ((10 shl 4) or 5).toByte()  // 0xA5
        // Second byte: low nibble = 3 (element 1), high nibble = 12 (element 33)
        codes[1] = ((12 shl 4) or 3).toByte()  // 0xC3

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        assertEquals(5, tensor.getCode(0, 0))    // lo byte 0
        assertEquals(3, tensor.getCode(0, 1))    // lo byte 1
        assertEquals(10, tensor.getCode(0, 32))  // hi byte 0
        assertEquals(12, tensor.getCode(0, 33))  // hi byte 1
    }

    @Test
    fun `Q4_KBlockTensorData handles all 4-bit values (canonical strided layout)`() {
        val codes = ByteArray(128)
        // Each byte's lo and hi nibble = (idx mod 16). With strided decoding,
        // element i (i<32) reads byte i lo, element i+32 reads byte i hi —
        // so a same-nibble byte means element i and element i+32 share value.
        for (i in 0 until 16) {
            codes[i] = ((i shl 4) or i).toByte()
        }

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        for (i in 0 until 16) {
            assertEquals(i, tensor.getCode(0, i),      "Code at strided lo index $i (byte $i lo)")
            assertEquals(i, tensor.getCode(0, i + 32), "Code at strided hi index ${i + 32} (byte $i hi)")
        }
    }

    @Test
    fun `Q4_KBlockTensorData get via indices works (canonical strided layout)`() {
        val codes = ByteArray(128)
        codes[0] = 0x21  // element 0 (lo nibble) = 1; element 32 (hi nibble) = 2

        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        assertEquals(1.toByte(), tensor[0])
        assertEquals(2.toByte(), tensor[32])
    }

    @Test
    fun `Q4_KBlockTensorData 2D access works correctly (canonical strided layout)`() {
        val codes = ByteArray(128)
        for (i in 0 until 128) {
            // strided: byte i in group `i/32` carries element (groupBase + i%32)
            // in lo and (groupBase + i%32 + 32) in hi.
            val groupBase = (i / 32) * 64
            val withinGroup = i % 32
            val lo = (groupBase + withinGroup) % 16
            val hi = (groupBase + withinGroup + 32) % 16
            codes[i] = ((hi shl 4) or lo).toByte()
        }

        // 16x16 = 256 elements = 1 block (row-major)
        val block = createQ4KBlock(0x3C00, 0x0000, codes = codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(16, 16), block)

        // tensor[0, 0] = element 0 = lo byte 0 = (0 + 0) % 16 = 0
        // tensor[0, 1] = element 1 = lo byte 1 = (0 + 1) % 16 = 1
        // tensor[2, 0] = element 32 = hi byte 0 = (0 + 32) % 16 = 0
        assertEquals(0, tensor[0, 0].toInt())
        assertEquals(1, tensor[0, 1].toInt())
        assertEquals(0, tensor[2, 0].toInt())
    }

    @Test
    fun `Q4_KBlockTensorData set operation works (canonical strided layout)`() {
        val block = createQ4KBlock(0x3C00, 0x0000)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        // Element 0 lives in byte 0 lo; element 32 lives in byte 0 hi.
        tensor[0] = 7
        tensor[32] = 11

        assertEquals(7.toByte(), tensor[0])
        assertEquals(11.toByte(), tensor[32])
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
    fun `Q4_KBlockTensorData toFloatArray produces expected values (canonical ggml formula)`() {
        // d = 1.0, dMin = 0.0. With ggml's `get_scale_min_k4` decoding of all
        // 0xFF scale bytes:
        //   sub-blocks 0..3: scaleIdx = 0x3F (low 6 of byte j)
        //   sub-blocks 4..7: scaleIdx = (low 4 of byte j+4) | (top 2 of byte j-4) << 4
        //                            = 0x0F | (0x03 << 4) = 0x3F
        // So all sub-blocks have scaleIdx = 63. With ggml's `scale = d * sc`
        // (no /63), the per-element scale is 1.0 * 63 = 63.0. Mins likewise.
        // With dMin = 0, offset = 0, so output[i] = code[i] * 63 - 0.
        val scaleMinIndices = ByteArray(12) { 0xFF.toByte() }

        val codes = ByteArray(128)
        codes[0] = 0x21  // element 0 (lo byte 0) = 1; element 32 (hi byte 0) = 2

        val block = createQ4KBlock(0x3C00, 0x0000, scaleMinIndices, codes)
        val tensor = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        val floats = tensor.toFloatArray()

        assertEquals(63.0f, floats[0], 0.1f)    // 1 * 63
        assertEquals(126.0f, floats[32], 0.1f)  // 2 * 63
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
