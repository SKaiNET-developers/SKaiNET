package sk.ainet.lang.tensor.ops.turboquant

/**
 * Bit-packing and unpacking for TurboQuant codes.
 *
 * Packs signed N-bit integer codes into compact byte arrays for storage.
 * Supports 2, 3, 4, and 8-bit packing. Codes are stored as unsigned
 * offsets (biased by 2^(bits-1)) to simplify packing.
 *
 * Packing is append-friendly: codes can be packed incrementally per token
 * without re-packing the entire cache.
 */
public object BitPacker {

    /**
     * Pack signed codes into a compact byte array.
     *
     * Codes are biased to unsigned range before packing:
     * stored = code + 2^(bits-1)
     *
     * @param codes Signed codes (values in [-maxCode, maxCode])
     * @param bits  Bits per code (2, 3, 4, or 8)
     * @return Packed byte array
     */
    public fun pack(codes: ByteArray, bits: Int): ByteArray {
        require(bits in setOf(2, 3, 4, 8)) { "bits must be 2, 3, 4, or 8, got $bits" }
        return when (bits) {
            2 -> pack2Bit(codes)
            3 -> pack3Bit(codes)
            4 -> pack4Bit(codes)
            8 -> pack8Bit(codes)
            else -> error("unreachable")
        }
    }

    /**
     * Unpack a byte array back to signed codes.
     *
     * @param packed  Packed byte array
     * @param count   Number of codes to unpack
     * @param bits    Bits per code (2, 3, 4, or 8)
     * @return Signed codes
     */
    public fun unpack(packed: ByteArray, count: Int, bits: Int): ByteArray {
        require(bits in setOf(2, 3, 4, 8)) { "bits must be 2, 3, 4, or 8, got $bits" }
        return when (bits) {
            2 -> unpack2Bit(packed, count)
            3 -> unpack3Bit(packed, count)
            4 -> unpack4Bit(packed, count)
            8 -> unpack8Bit(packed, count)
            else -> error("unreachable")
        }
    }

    /**
     * Compute the byte size needed to pack [count] codes at [bits] per code.
     */
    public fun packedSize(count: Int, bits: Int): Int {
        return when (bits) {
            2 -> (count + 3) / 4
            3 -> (count * 3 + 7) / 8
            4 -> (count + 1) / 2
            8 -> count
            else -> throw IllegalArgumentException("bits must be 2, 3, 4, or 8")
        }
    }

    // ========== 2-bit packing ==========
    // 4 codes per byte. Bias = 2 (range: [-1,1] → [1,3], stored as [0,3])

    private fun pack2Bit(codes: ByteArray): ByteArray {
        val bias = 2 // 2^(2-1)
        val packed = ByteArray((codes.size + 3) / 4)
        for (i in codes.indices) {
            val unsigned = (codes[i].toInt() + bias) and 0x03
            val byteIdx = i / 4
            val shift = (i % 4) * 2
            packed[byteIdx] = (packed[byteIdx].toInt() or (unsigned shl shift)).toByte()
        }
        return packed
    }

    private fun unpack2Bit(packed: ByteArray, count: Int): ByteArray {
        val bias = 2
        val codes = ByteArray(count)
        for (i in 0 until count) {
            val byteIdx = i / 4
            val shift = (i % 4) * 2
            val unsigned = (packed[byteIdx].toInt() ushr shift) and 0x03
            codes[i] = (unsigned - bias).toByte()
        }
        return codes
    }

    // ========== 3-bit packing ==========
    // 8 codes per 3 bytes. Bias = 4 (range: [-3,3] → [1,7], stored as [0,7])

    private fun pack3Bit(codes: ByteArray): ByteArray {
        val bias = 4 // 2^(3-1)
        val packed = ByteArray((codes.size * 3 + 7) / 8)
        var bitPos = 0
        for (i in codes.indices) {
            val unsigned = (codes[i].toInt() + bias) and 0x07
            val byteIdx = bitPos / 8
            val bitOffset = bitPos % 8
            packed[byteIdx] = (packed[byteIdx].toInt() or (unsigned shl bitOffset)).toByte()
            // Handle overflow into next byte
            if (bitOffset > 5) {
                val overflow = unsigned ushr (8 - bitOffset)
                if (byteIdx + 1 < packed.size) {
                    packed[byteIdx + 1] = (packed[byteIdx + 1].toInt() or overflow).toByte()
                }
            }
            bitPos += 3
        }
        return packed
    }

    private fun unpack3Bit(packed: ByteArray, count: Int): ByteArray {
        val bias = 4
        val codes = ByteArray(count)
        var bitPos = 0
        for (i in 0 until count) {
            val byteIdx = bitPos / 8
            val bitOffset = bitPos % 8
            var value = (packed[byteIdx].toInt() ushr bitOffset) and 0x07
            // Handle cross-byte boundary
            if (bitOffset > 5 && byteIdx + 1 < packed.size) {
                val bitsFromFirst = 8 - bitOffset
                val remaining = 3 - bitsFromFirst
                val fromNext = packed[byteIdx + 1].toInt() and ((1 shl remaining) - 1)
                value = ((packed[byteIdx].toInt() ushr bitOffset) and ((1 shl bitsFromFirst) - 1)) or
                        (fromNext shl bitsFromFirst)
            }
            codes[i] = (value - bias).toByte()
            bitPos += 3
        }
        return codes
    }

    // ========== 4-bit packing ==========
    // 2 codes per byte. Bias = 8 (range: [-7,7] → [1,15], stored as [0,15])

    private fun pack4Bit(codes: ByteArray): ByteArray {
        val bias = 8 // 2^(4-1)
        val packed = ByteArray((codes.size + 1) / 2)
        for (i in codes.indices) {
            val unsigned = (codes[i].toInt() + bias) and 0x0F
            val byteIdx = i / 2
            if (i % 2 == 0) {
                packed[byteIdx] = (packed[byteIdx].toInt() or unsigned).toByte()
            } else {
                packed[byteIdx] = (packed[byteIdx].toInt() or (unsigned shl 4)).toByte()
            }
        }
        return packed
    }

    private fun unpack4Bit(packed: ByteArray, count: Int): ByteArray {
        val bias = 8
        val codes = ByteArray(count)
        for (i in 0 until count) {
            val byteIdx = i / 2
            val unsigned = if (i % 2 == 0) {
                packed[byteIdx].toInt() and 0x0F
            } else {
                (packed[byteIdx].toInt() ushr 4) and 0x0F
            }
            codes[i] = (unsigned - bias).toByte()
        }
        return codes
    }

    // ========== 8-bit packing ==========
    // 1:1 mapping, codes are already bytes

    private fun pack8Bit(codes: ByteArray): ByteArray = codes.copyOf()

    private fun unpack8Bit(packed: ByteArray, count: Int): ByteArray {
        return if (packed.size == count) packed.copyOf()
        else packed.copyOfRange(0, count)
    }
}
