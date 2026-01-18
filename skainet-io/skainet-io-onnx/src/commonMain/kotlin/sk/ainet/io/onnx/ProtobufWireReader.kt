package sk.ainet.io.onnx

import sk.ainet.io.RandomAccessSource

/**
 * Low-level Protocol Buffer wire format reader with random access support.
 *
 * Enables parsing protobuf messages without loading entire file into memory.
 * Supports skipping large fields (like tensor raw_data) and recording their
 * positions for lazy loading.
 *
 * Wire format reference: https://protobuf.dev/programming-guides/encoding/
 */
internal class ProtobufWireReader(
    private val source: RandomAccessSource
) {
    /** Current read position in the file */
    var position: Long = 0L
        private set

    /** File size for bounds checking */
    val size: Long get() = source.size

    /** Check if more data is available */
    fun hasRemaining(): Boolean = position < source.size

    /** Check if more data is available within a limit */
    fun hasRemaining(limit: Long): Boolean = position < limit && position < source.size

    /**
     * Read a varint (variable-length integer) from current position.
     * Used for field tags, int32, int64, uint32, uint64, sint32, sint64, bool, enum.
     */
    fun readVarint(): Long {
        var result = 0L
        var shift = 0
        while (true) {
            val b = source.readByteAt(position++).toInt() and 0xFF
            result = result or ((b and 0x7F).toLong() shl shift)
            if (b and 0x80 == 0) break
            shift += 7
            if (shift >= 64) throw IllegalStateException("Varint too long")
        }
        return result
    }

    /**
     * Read a fixed 32-bit value (little-endian).
     * Used for fixed32, sfixed32, float.
     */
    fun readFixed32(): Int {
        val bytes = source.readAt(position, 4)
        position += 4
        return (bytes[0].toInt() and 0xFF) or
                ((bytes[1].toInt() and 0xFF) shl 8) or
                ((bytes[2].toInt() and 0xFF) shl 16) or
                ((bytes[3].toInt() and 0xFF) shl 24)
    }

    /**
     * Read a fixed 64-bit value (little-endian).
     * Used for fixed64, sfixed64, double.
     */
    fun readFixed64(): Long {
        val bytes = source.readAt(position, 8)
        position += 8
        return (bytes[0].toLong() and 0xFF) or
                ((bytes[1].toLong() and 0xFF) shl 8) or
                ((bytes[2].toLong() and 0xFF) shl 16) or
                ((bytes[3].toLong() and 0xFF) shl 24) or
                ((bytes[4].toLong() and 0xFF) shl 32) or
                ((bytes[5].toLong() and 0xFF) shl 40) or
                ((bytes[6].toLong() and 0xFF) shl 48) or
                ((bytes[7].toLong() and 0xFF) shl 56)
    }

    /**
     * Read length-delimited bytes (string, bytes, embedded message, packed repeated).
     * Returns the byte array.
     */
    fun readBytes(): ByteArray {
        val length = readVarint().toInt()
        if (length < 0) throw IllegalStateException("Negative length: $length")
        if (length == 0) return ByteArray(0)
        val bytes = source.readAt(position, length)
        position += length
        return bytes
    }

    /**
     * Read a length-delimited string.
     */
    fun readString(): String = readBytes().decodeToString()

    /**
     * Skip a length-delimited field without loading its content.
     * Returns the (startPosition, length) for lazy loading.
     */
    fun skipBytes(): Pair<Long, Int> {
        val length = readVarint().toInt()
        if (length < 0) throw IllegalStateException("Negative length: $length")
        val startPos = position
        position += length
        return startPos to length
    }

    /**
     * Skip a field based on its wire type.
     * @param wireType The wire type (0-5)
     */
    fun skipField(wireType: Int) {
        when (wireType) {
            WIRE_TYPE_VARINT -> readVarint()
            WIRE_TYPE_FIXED64 -> position += 8
            WIRE_TYPE_LENGTH_DELIMITED -> {
                val length = readVarint().toInt()
                position += length
            }
            WIRE_TYPE_START_GROUP -> throw UnsupportedOperationException("Groups not supported")
            WIRE_TYPE_END_GROUP -> { /* nothing to skip */ }
            WIRE_TYPE_FIXED32 -> position += 4
            else -> throw IllegalStateException("Unknown wire type: $wireType")
        }
    }

    /**
     * Seek to a specific position.
     */
    fun seek(newPosition: Long) {
        require(newPosition >= 0 && newPosition <= source.size) {
            "Position out of bounds: $newPosition (size: ${source.size})"
        }
        position = newPosition
    }

    /**
     * Read bytes from a specific position without changing current position.
     */
    fun readBytesAt(pos: Long, length: Int): ByteArray {
        return source.readAt(pos, length)
    }

    companion object {
        // Wire types
        const val WIRE_TYPE_VARINT = 0
        const val WIRE_TYPE_FIXED64 = 1
        const val WIRE_TYPE_LENGTH_DELIMITED = 2
        const val WIRE_TYPE_START_GROUP = 3
        const val WIRE_TYPE_END_GROUP = 4
        const val WIRE_TYPE_FIXED32 = 5

        /** Extract field number from tag */
        fun fieldNumber(tag: Long): Int = (tag shr 3).toInt()

        /** Extract wire type from tag */
        fun wireType(tag: Long): Int = (tag and 0x07).toInt()
    }
}
