package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals
import kotlin.math.abs

class Q8_0TensorDataTest {

    @Test
    fun `Q8_0BlockTensorData can read scale from block`() {
        // Create one block: 2 bytes f16 scale + 32 bytes codes
        val blockData = ByteArray(34)

        // Scale = 1.0 (f16 0x3C00 little-endian)
        blockData[0] = 0x00
        blockData[1] = 0x3C

        // Fill codes with values 0-31
        for (i in 0 until 32) {
            blockData[2 + i] = i.toByte()
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)

        assertEquals(1.0f, tensor.getBlockScale(0), 0.001f)
    }

    @Test
    fun `Q8_0BlockTensorData can read codes from block`() {
        val blockData = ByteArray(34)

        // Scale = 0.5 (f16 0x3800 little-endian)
        blockData[0] = 0x00
        blockData[1] = 0x38

        // Fill codes with known values
        for (i in 0 until 32) {
            blockData[2 + i] = (i - 16).toByte()  // -16 to +15
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)

        for (i in 0 until 32) {
            val expected = (i - 16).toByte()
            assertEquals(expected, tensor.getCode(0, i), "Code at index $i mismatch")
        }
    }

    @Test
    fun `Q8_0BlockTensorData get via indices works`() {
        val blockData = ByteArray(34)
        blockData[0] = 0x00
        blockData[1] = 0x3C  // scale = 1.0

        for (i in 0 until 32) {
            blockData[2 + i] = (i * 2).toByte()
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)

        for (i in 0 until 32) {
            assertEquals((i * 2).toByte(), tensor[i])
        }
    }

    @Test
    fun `Q8_0BlockTensorData 2D access works correctly`() {
        // 4x8 = 32 elements = 1 block
        val blockData = ByteArray(34)
        blockData[0] = 0x00
        blockData[1] = 0x3C  // scale = 1.0

        for (i in 0 until 32) {
            blockData[2 + i] = i.toByte()
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(4, 8), blockData)

        // tensor[0, 0] = element 0
        // tensor[0, 7] = element 7
        // tensor[1, 0] = element 8
        // tensor[3, 7] = element 31
        assertEquals(0.toByte(), tensor[0, 0])
        assertEquals(7.toByte(), tensor[0, 7])
        assertEquals(8.toByte(), tensor[1, 0])
        assertEquals(31.toByte(), tensor[3, 7])
    }

    @Test
    fun `Q8_0BlockTensorData toFloatArray applies scale correctly`() {
        val blockData = ByteArray(34)

        // Scale = 0.5
        blockData[0] = 0x00
        blockData[1] = 0x38

        // Codes: 2, -2, 4, -4
        blockData[2] = 2
        blockData[3] = (-2).toByte()
        blockData[4] = 4
        blockData[5] = (-4).toByte()
        for (i in 6 until 34) {
            blockData[i] = 0
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)
        val floats = tensor.toFloatArray()

        assertEquals(1.0f, floats[0], 0.01f)   // 2 * 0.5
        assertEquals(-1.0f, floats[1], 0.01f)  // -2 * 0.5
        assertEquals(2.0f, floats[2], 0.01f)   // 4 * 0.5
        assertEquals(-2.0f, floats[3], 0.01f)  // -4 * 0.5
    }

    @Test
    fun `Q8_0BlockTensorData handles multiple blocks`() {
        // 64 elements = 2 blocks
        val blockData = ByteArray(68)

        // Block 0: scale = 1.0, codes = 0
        blockData[0] = 0x00
        blockData[1] = 0x3C
        for (i in 2 until 34) {
            blockData[i] = 10
        }

        // Block 1: scale = 2.0, codes = 5
        blockData[34] = 0x00
        blockData[35] = 0x40  // f16 2.0 = 0x4000
        for (i in 36 until 68) {
            blockData[i] = 5
        }

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(64), blockData)

        assertEquals(2, tensor.blockCount)
        assertEquals(1.0f, tensor.getBlockScale(0), 0.001f)
        assertEquals(2.0f, tensor.getBlockScale(1), 0.001f)

        // First block elements
        assertEquals(10.toByte(), tensor.getCode(0, 0))
        assertEquals(10.toByte(), tensor.getCode(0, 31))

        // Second block elements
        assertEquals(5.toByte(), tensor.getCode(1, 0))
        assertEquals(5.toByte(), tensor.getCode(1, 31))
    }

    @Test
    fun `Q8_0BlockTensorData set operation works`() {
        val blockData = ByteArray(34)
        blockData[0] = 0x00
        blockData[1] = 0x3C

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)

        tensor[5] = 42
        tensor[10] = (-10).toByte()

        assertEquals(42.toByte(), tensor[5])
        assertEquals((-10).toByte(), tensor[10])
    }

    @Test
    fun `Q8_0BlockTensorData shape is preserved`() {
        // 8x4 = 32 elements = 1 block
        val blockData = ByteArray(34)
        blockData[0] = 0x00
        blockData[1] = 0x3C

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(8, 4), blockData)

        assertContentEquals(intArrayOf(8, 4), tensor.shape.dimensions)
        assertEquals(32, tensor.shape.volume)
        assertEquals(1, tensor.blockCount)
    }

    @Test
    fun `Q8_0BlockTensorData handles negative scale`() {
        val blockData = ByteArray(34)

        // Scale = -0.5 (f16 0xB800)
        blockData[0] = 0x00
        blockData[1] = 0xB8.toByte()

        blockData[2] = 4

        val tensor = Q8_0BlockTensorData.fromRawBytes(Shape(32), blockData)
        val floats = tensor.toFloatArray()

        assertEquals(-2.0f, floats[0], 0.01f)  // 4 * -0.5
    }

    @Test
    fun `Q8_0TensorData constants are correct`() {
        assertEquals(32, Q8_0TensorData.BLOCK_SIZE)
        assertEquals(34, Q8_0TensorData.BYTES_PER_BLOCK)
    }
}
