package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals

class TernaryTensorDataTest {

    @Test
    fun `Ternary2BitTensorData zeros returns all zeros`() {
        val tensor = Ternary2BitTensorData.zeros(Shape(8))

        for (i in 0 until 8) {
            assertEquals(0.toByte(), tensor[i], "Element $i should be 0")
        }
    }

    @Test
    fun `Ternary2BitTensorData can store and retrieve ternary values`() {
        val values = byteArrayOf(-1, 0, 1, -1, 1, 0, 0, 1)
        val tensor = Ternary2BitTensorData.fromTernaryValues(Shape(8), values)

        for (i in values.indices) {
            assertEquals(values[i], tensor[i], "Element $i mismatch")
        }
    }

    @Test
    fun `Ternary2BitTensorData 2D access works correctly`() {
        // 2x4 tensor
        val values = byteArrayOf(-1, 0, 1, -1, 1, 0, 0, 1)
        val tensor = Ternary2BitTensorData.fromTernaryValues(Shape(2, 4), values)

        // Row 0: -1, 0, 1, -1
        assertEquals((-1).toByte(), tensor[0, 0])
        assertEquals(0.toByte(), tensor[0, 1])
        assertEquals(1.toByte(), tensor[0, 2])
        assertEquals((-1).toByte(), tensor[0, 3])

        // Row 1: 1, 0, 0, 1
        assertEquals(1.toByte(), tensor[1, 0])
        assertEquals(0.toByte(), tensor[1, 1])
        assertEquals(0.toByte(), tensor[1, 2])
        assertEquals(1.toByte(), tensor[1, 3])
    }

    @Test
    fun `Ternary2BitTensorData set operation works`() {
        val tensor = Ternary2BitTensorData.zeros(Shape(4))

        tensor[0] = -1
        tensor[1] = 1
        tensor[2] = 0
        tensor[3] = 1

        assertEquals((-1).toByte(), tensor[0])
        assertEquals(1.toByte(), tensor[1])
        assertEquals(0.toByte(), tensor[2])
        assertEquals(1.toByte(), tensor[3])
    }

    @Test
    fun `Ternary2BitTensorData toFloatArray applies scale`() {
        val values = byteArrayOf(-1, 0, 1, -1)
        val tensor = Ternary2BitTensorData.fromTernaryValues(Shape(4), values, scale = 2.0f)

        val floats = tensor.toFloatArray()

        assertEquals(-2.0f, floats[0], 0.001f)
        assertEquals(0.0f, floats[1], 0.001f)
        assertEquals(2.0f, floats[2], 0.001f)
        assertEquals(-2.0f, floats[3], 0.001f)
    }

    @Test
    fun `Ternary2BitTensorData all minus ones decodes correctly`() {
        // Packed data: 0x00 = 00 00 00 00 = all zeros which decode to -1
        val packedData = byteArrayOf(0x00)
        val tensor = Ternary2BitTensorData(Shape(4), packedData)

        for (i in 0 until 4) {
            assertEquals((-1).toByte(), tensor[i], "Element $i should be -1")
        }
    }

    @Test
    fun `Ternary2BitTensorData all zeros encodes as 0x55`() {
        // TQ2_0 encoding: 1 = 0 (ternary zero)
        // 4 zeros: 01 01 01 01 = 0x55
        val packedData = byteArrayOf(0x55)
        val tensor = Ternary2BitTensorData(Shape(4), packedData)

        for (i in 0 until 4) {
            assertEquals(0.toByte(), tensor[i], "Element $i should be 0")
        }
    }

    @Test
    fun `Ternary2BitTensorData all plus ones encodes as 0xAA`() {
        // TQ2_0 encoding: 2 = +1
        // 4 +1s: 10 10 10 10 = 0xAA
        val packedData = byteArrayOf(0xAA.toByte())
        val tensor = Ternary2BitTensorData(Shape(4), packedData)

        for (i in 0 until 4) {
            assertEquals(1.toByte(), tensor[i], "Element $i should be +1")
        }
    }

    @Test
    fun `Ternary2BitTensorData handles larger tensors`() {
        // 256 elements = 64 bytes of packed data
        val values = ByteArray(256) { i -> ((i % 3) - 1).toByte() }  // -1, 0, +1 pattern
        val tensor = Ternary2BitTensorData.fromTernaryValues(Shape(256), values)

        for (i in values.indices) {
            assertEquals(values[i], tensor[i], "Element $i mismatch")
        }
    }

    @Test
    fun `Ternary2BitTensorData fromTQ2_0Block decodes block correctly`() {
        // Create a TQ2_0 block: 64 bytes data + 2 bytes scale
        val blockData = ByteArray(66)

        // All zeros in data means all -1 in ternary
        for (i in 0 until 64) {
            blockData[i] = 0x00
        }

        // Scale = 1.0 (f16 0x3C00 little-endian)
        blockData[64] = 0x00
        blockData[65] = 0x3C

        val tensor = Ternary2BitTensorData.fromTQ2_0Block(blockData, Shape(256))

        assertEquals(1.0f, tensor.scale, 0.001f)

        for (i in 0 until 256) {
            assertEquals((-1).toByte(), tensor[i], "Element $i should be -1")
        }
    }

    @Test
    fun `Ternary2BitTensorData shape is preserved`() {
        val tensor = Ternary2BitTensorData.zeros(Shape(4, 8, 2))

        assertContentEquals(intArrayOf(4, 8, 2), tensor.shape.dimensions)
        assertEquals(64, tensor.shape.volume)
    }

    @Test
    fun `TernaryTensorData interface scale accessor works`() {
        val tensor: TernaryTensorData = Ternary2BitTensorData.fromTernaryValues(
            Shape(4),
            byteArrayOf(1, 0, -1, 0),
            scale = 0.5f
        )

        assertEquals(0.5f, tensor.scale, 0.001f)
    }

    @Test
    fun `TernaryTensorData interface packedData accessor works`() {
        val values = byteArrayOf(-1, 0, 1, -1)  // 00 01 10 00 = 0x24
        val tensor: TernaryTensorData = Ternary2BitTensorData.fromTernaryValues(Shape(4), values)

        // Verify packed data is accessible
        val packed = tensor.packedData
        assertEquals(1, packed.size)
    }
}
