package sk.ainet.lang.tensor.ops.turboquant

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BitPackerTest {

    @Test
    fun pack4BitRoundTrip() {
        val codes = byteArrayOf(0, 1, -1, 7, -7, 3, -3, 0)
        val packed = BitPacker.pack(codes, 4)
        val unpacked = BitPacker.unpack(packed, codes.size, 4)

        assertTrue(codes.contentEquals(unpacked),
            "4-bit round trip failed: ${codes.toList()} -> ${unpacked.toList()}")
    }

    @Test
    fun pack2BitRoundTrip() {
        val codes = byteArrayOf(0, 1, -1, 0, 1, -1, 0, 1)
        val packed = BitPacker.pack(codes, 2)
        val unpacked = BitPacker.unpack(packed, codes.size, 2)

        assertTrue(codes.contentEquals(unpacked),
            "2-bit round trip failed: ${codes.toList()} -> ${unpacked.toList()}")
    }

    @Test
    fun pack3BitRoundTrip() {
        val codes = byteArrayOf(0, 1, -1, 3, -3, 2, -2, 0)
        val packed = BitPacker.pack(codes, 3)
        val unpacked = BitPacker.unpack(packed, codes.size, 3)

        assertTrue(codes.contentEquals(unpacked),
            "3-bit round trip failed: ${codes.toList()} -> ${unpacked.toList()}")
    }

    @Test
    fun pack8BitRoundTrip() {
        val codes = byteArrayOf(0, 127, -128, 1, -1, 64, -64, 100)
        val packed = BitPacker.pack(codes, 8)
        val unpacked = BitPacker.unpack(packed, codes.size, 8)

        assertTrue(codes.contentEquals(unpacked))
    }

    @Test
    fun pack4BitCompression() {
        val codes = ByteArray(100)
        val packed = BitPacker.pack(codes, 4)
        assertEquals(50, packed.size, "4-bit should be 50% size")
    }

    @Test
    fun pack2BitCompression() {
        val codes = ByteArray(100)
        val packed = BitPacker.pack(codes, 2)
        assertEquals(25, packed.size, "2-bit should be 25% size")
    }

    @Test
    fun packedSize() {
        assertEquals(50, BitPacker.packedSize(100, 4))
        assertEquals(25, BitPacker.packedSize(100, 2))
        assertEquals(100, BitPacker.packedSize(100, 8))
        assertEquals(38, BitPacker.packedSize(100, 3)) // (100*3+7)/8
    }

    @Test
    fun oddCountRoundTrip() {
        // Non-aligned count
        val codes = byteArrayOf(1, -1, 0)
        val packed4 = BitPacker.pack(codes, 4)
        val unpacked4 = BitPacker.unpack(packed4, 3, 4)
        assertTrue(codes.contentEquals(unpacked4))

        val packed2 = BitPacker.pack(codes, 2)
        val unpacked2 = BitPacker.unpack(packed2, 3, 2)
        // 2-bit can only represent -1, 0, 1 — codes[0]=1, codes[1]=-1, codes[2]=0 all valid
        assertTrue(codes.contentEquals(unpacked2))
    }

    @Test
    fun pack4BitAllValues() {
        // Test all valid 4-bit values: -7 to 7
        val codes = ByteArray(15) { (it - 7).toByte() }
        val packed = BitPacker.pack(codes, 4)
        val unpacked = BitPacker.unpack(packed, 15, 4)
        assertTrue(codes.contentEquals(unpacked),
            "All 4-bit values: ${codes.toList()} -> ${unpacked.toList()}")
    }
}
