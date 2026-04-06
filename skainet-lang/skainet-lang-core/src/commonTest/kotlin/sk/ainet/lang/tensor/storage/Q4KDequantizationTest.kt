package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Q4KDequantizationTest {

    /**
     * Build a 144-byte Q4_K block with controlled values.
     *
     * Layout:
     * - bytes [0..1]: f16 d (main scale)
     * - bytes [2..3]: f16 dMin (minimum scale)
     * - bytes [4..15]: packed 12-bit scale/min indices (12 bytes)
     * - bytes [16..143]: 4-bit codes (128 bytes, 2 codes per byte)
     */
    private fun buildQ4KBlock(
        d: Float = 1.0f,
        dMin: Float = 0.0f,
        codeValue: Int = 0
    ): ByteArray {
        val block = ByteArray(Q4_KTensorData.BYTES_PER_BLOCK) // 144

        // d as f16 little-endian
        val dBits = floatToHalf(d)
        block[0] = (dBits and 0xFF).toByte()
        block[1] = ((dBits shr 8) and 0xFF).toByte()

        // dMin as f16 little-endian
        val dMinBits = floatToHalf(dMin)
        block[2] = (dMinBits and 0xFF).toByte()
        block[3] = ((dMinBits shr 8) and 0xFF).toByte()

        // Scale/min indices: all 63 for scale, all 0 for min
        // Each sub-block uses 12 bits: 6 for scaleIdx + 6 for minIdx
        // 8 sub-blocks * 12 bits = 96 bits = 12 bytes
        // scaleIdx=63 (0x3F), minIdx=0 (0x00) → 12 bits per sub-block = 0xFC0 → little-endian
        for (i in 0 until 12) {
            // Pack all scale indices as 63 and min indices as 0
            // Bit pattern per sub-block: scaleIdx=111111, minIdx=000000
            // In 12-bit groups: 0b111111_000000 = 0xFC0
            val bitStart = i * 8
            var byteVal = 0
            for (bit in 0 until 8) {
                val globalBit = bitStart + bit
                val subBlock = globalBit / 12
                val bitInSubBlock = globalBit % 12
                if (subBlock < 8 && bitInSubBlock < 6) {
                    // This is a scale index bit — set to 1 (index = 63)
                    byteVal = byteVal or (1 shl bit)
                }
                // min index bits stay 0
            }
            block[4 + i] = byteVal.toByte()
        }

        // 4-bit codes: fill all with codeValue (0..15)
        val codeByte = ((codeValue and 0x0F) or ((codeValue and 0x0F) shl 4)).toByte()
        for (i in 16 until 144) {
            block[i] = codeByte
        }

        return block
    }

    private fun floatToHalf(value: Float): Int {
        val bits = value.toRawBits()
        val sign = (bits shr 16) and 0x8000
        val exponent = ((bits shr 23) and 0xFF) - 127
        val mantissa = bits and 0x7FFFFF

        return when {
            exponent >= 16 -> sign or 0x7C00 // overflow → infinity
            exponent >= -14 -> sign or ((exponent + 15) shl 10) or (mantissa shr 13)
            else -> sign // underflow → zero
        }
    }

    @Test
    fun dequantizeBlock_uniformCodes_producesExpectedOutput() {
        // d=1.0, dMin=0.0, all scale indices=63, all codes=5
        // scale = d * (63/63) = 1.0, min = 0.0
        // output = code * scale + min = 5 * 1.0 + 0.0 = 5.0
        val block = buildQ4KBlock(d = 1.0f, dMin = 0.0f, codeValue = 5)
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        val output = FloatArray(256)
        td.dequantizeBlock(0, output)

        for (i in 0 until 256) {
            assertEquals(5.0f, output[i], "Element $i should be 5.0")
        }
    }

    @Test
    fun getCode_lowAndHighNibble_correct() {
        val block = ByteArray(144)
        // Put a known byte at code position: byte at offset 16
        // Low nibble = 0xA (10), high nibble = 0x5 (5)
        block[16] = 0x5A.toByte()

        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)
        // Element 0 → low nibble of byte 16 → 0xA = 10
        assertEquals(10, td.getCode(0, 0))
        // Element 1 → high nibble of byte 16 → 0x5 = 5
        assertEquals(5, td.getCode(0, 1))
    }

    @Test
    fun toFloatArray_multiBlock_concatenatesBlocks() {
        // 2 blocks = 512 elements
        val data = ByteArray(288) // 2 * 144
        // Both blocks: d=1.0, dMin=0.0, all codes=0
        val block1 = buildQ4KBlock(d = 1.0f, dMin = 0.0f, codeValue = 0)
        block1.copyInto(data, 0)
        block1.copyInto(data, 144)

        val td = Q4_KBlockTensorData.fromRawBytes(Shape(512), data)
        val floats = (td as PackedBlockStorage).toFloatArray()

        assertEquals(512, floats.size)
    }

    @Test
    fun dequantizeBlock_outOfBoundsIndex_throws() {
        val block = ByteArray(144)
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)
        val output = FloatArray(256)

        assertFailsWith<IllegalArgumentException> {
            td.dequantizeBlock(-1, output)
        }
        assertFailsWith<IllegalArgumentException> {
            td.dequantizeBlock(1, output) // only 1 block (index 0)
        }
    }

    @Test
    fun physicalBytes_matchesExpected() {
        val block = ByteArray(144)
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)
        val packed = td as PackedBlockStorage

        assertEquals(144L, packed.physicalBytes)
        assertEquals(256L, packed.elementCount)
        assertEquals(1, packed.blockCount)
        assertEquals(256, packed.blockSize)
    }

    @Test
    fun dequantizeBlock_zeroCodes_producesMinValues() {
        // d=1.0, dMin=0.0, all codes=0 → output = 0*scale + min = 0.0
        val block = buildQ4KBlock(d = 1.0f, dMin = 0.0f, codeValue = 0)
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), block)

        val output = FloatArray(256)
        td.dequantizeBlock(0, output)

        for (i in 0 until 256) {
            assertEquals(0.0f, output[i], "Element $i should be 0.0 for zero codes")
        }
    }
}
