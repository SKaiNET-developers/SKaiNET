package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.tensor.data.toFloatArray
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class TernaryDequantizationTest {

    @Test
    fun dequantizeBlock_allMinusOnes_producesNegativeScale() {
        // Encoding: 0→-1, so 0x00 = four -1 values per byte
        val packed = ByteArray(2) { 0x00 } // 8 elements, all -1
        val td = Ternary2BitTensorData(Shape(8), packed, scale = 2.0f)
        val ps = td as PackedBlockStorage

        val output = FloatArray(8)
        ps.dequantizeBlock(0, output)

        for (i in 0 until 8) {
            assertEquals(-2.0f, output[i], "Element $i should be -1 * 2.0 = -2.0")
        }
    }

    @Test
    fun dequantizeBlock_allZeros_producesZeros() {
        // Encoding: 1→0, so 0x55 = 01_01_01_01 = four 0 values per byte
        val packed = ByteArray(2) { 0x55 }
        val td = Ternary2BitTensorData(Shape(8), packed, scale = 5.0f)
        val ps = td as PackedBlockStorage

        val output = FloatArray(8)
        ps.dequantizeBlock(0, output)

        for (i in 0 until 8) {
            assertEquals(0.0f, output[i], "Element $i should be 0 * 5.0 = 0.0")
        }
    }

    @Test
    fun dequantizeBlock_allPlusOnes_producesPositiveScale() {
        // Encoding: 2→+1, so 0xAA = 10_10_10_10 = four +1 values per byte
        val packed = ByteArray(2) { 0xAA.toByte() }
        val td = Ternary2BitTensorData(Shape(8), packed, scale = 3.0f)
        val ps = td as PackedBlockStorage

        val output = FloatArray(8)
        ps.dequantizeBlock(0, output)

        for (i in 0 until 8) {
            assertEquals(3.0f, output[i], "Element $i should be +1 * 3.0 = 3.0")
        }
    }

    @Test
    fun dequantizeBlock_mixedValues_matchesToFloatArray() {
        // Mixed: -1, 0, +1, -1 per byte → 0b10_01_00 = 0x00+bits
        // Byte: bits [1:0]=00(-1), [3:2]=01(0), [5:4]=10(+1), [7:6]=00(-1)
        // = 0b00_10_01_00 = 0x24
        val packed = byteArrayOf(0x24, 0x24)
        val td = Ternary2BitTensorData(Shape(8), packed, scale = 1.0f)

        // Verify via PackedBlockStorage
        val ps = td as PackedBlockStorage
        val output = FloatArray(8)
        ps.dequantizeBlock(0, output)

        // Also verify via extension function
        val expected = td.toFloatArray()

        for (i in 0 until 8) {
            assertEquals(expected[i], output[i], "Element $i: dequantizeBlock should match toFloatArray")
        }
    }

    @Test
    fun dequantizeBlock_withOutputOffset_writesAtOffset() {
        val packed = ByteArray(1) { 0xAA.toByte() } // 4 elements, all +1
        val td = Ternary2BitTensorData(Shape(4), packed, scale = 1.0f)
        val ps = td as PackedBlockStorage

        val output = FloatArray(14) // larger than needed
        ps.dequantizeBlock(0, output, outputOffset = 10)

        // Elements [0..9] should be untouched (0.0)
        for (i in 0 until 10) {
            assertEquals(0.0f, output[i], "Element $i should be untouched")
        }
        // Elements [10..13] should be 1.0
        for (i in 10 until 14) {
            assertEquals(1.0f, output[i], "Element $i should be 1.0")
        }
    }

    @Test
    fun dequantizeBlock_invalidBlockIndex_throws() {
        val packed = ByteArray(1) { 0x55 }
        val td = Ternary2BitTensorData(Shape(4), packed) as PackedBlockStorage

        assertFailsWith<IllegalArgumentException> {
            td.dequantizeBlock(1, FloatArray(4)) // only block 0 valid
        }
    }
}
