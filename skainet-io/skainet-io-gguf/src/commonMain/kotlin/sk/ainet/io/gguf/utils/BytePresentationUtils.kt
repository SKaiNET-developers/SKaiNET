package sk.ainet.io.gguf.utils

import kotlin.reflect.KClass


enum class Endian {
    BIG_ENDIAN, LITTLE_ENDIAN
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toUInt(endian: Endian): UInt {
    require(size >= 4) { "ByteArray size must be at least 4 to convert to UInt" }
    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toUInt() and 0xFFu) shl 24) or
                ((this[1].toUInt() and 0xFFu) shl 16) or
                ((this[2].toUInt() and 0xFFu) shl 8) or
                (this[3].toUInt() and 0xFFu)

        Endian.LITTLE_ENDIAN -> ((this[3].toUInt() and 0xFFu) shl 24) or
                ((this[2].toUInt() and 0xFFu) shl 16) or
                ((this[1].toUInt() and 0xFFu) shl 8) or
                (this[0].toUInt() and 0xFFu)
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toULong(endian: Endian): ULong {
    require(size >= 8) { "ByteArray size must be at least 8 to convert to ULong" }

    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toULong() and 0xFFuL) shl 56) or
                ((this[1].toULong() and 0xFFuL) shl 48) or
                ((this[2].toULong() and 0xFFuL) shl 40) or
                ((this[3].toULong() and 0xFFuL) shl 32) or
                ((this[4].toULong() and 0xFFuL) shl 24) or
                ((this[5].toULong() and 0xFFuL) shl 16) or
                ((this[6].toULong() and 0xFFuL) shl 8) or
                (this[7].toULong() and 0xFFuL)

        Endian.LITTLE_ENDIAN -> ((this[7].toULong() and 0xFFuL) shl 56) or
                ((this[6].toULong() and 0xFFuL) shl 48) or
                ((this[5].toULong() and 0xFFuL) shl 40) or
                ((this[4].toULong() and 0xFFuL) shl 32) or
                ((this[3].toULong() and 0xFFuL) shl 24) or
                ((this[2].toULong() and 0xFFuL) shl 16) or
                ((this[1].toULong() and 0xFFuL) shl 8) or
                (this[0].toULong() and 0xFFuL)
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toInt(endian: Endian): Int {
    require(size >= 4) { "ByteArray size must be at least 4 to convert to Int" }

    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toInt() and 0xFF) shl 24) or
                ((this[1].toInt() and 0xFF) shl 16) or
                ((this[2].toInt() and 0xFF) shl 8) or
                (this[3].toInt() and 0xFF)

        Endian.LITTLE_ENDIAN -> ((this[3].toInt() and 0xFF) shl 24) or
                ((this[2].toInt() and 0xFF) shl 16) or
                ((this[1].toInt() and 0xFF) shl 8) or
                (this[0].toInt() and 0xFF)
    }
}


/** Converts a List<UByte> (of size 4) to a UInt, taking endianness into account. */
fun List<UByte>.toUInt(endian: Endian): UInt {
    require(size == 4) { "Expected 4 bytes for UInt, got $size" }
    return if (endian == Endian.LITTLE_ENDIAN)
        (this[0].toUInt()
                or (this[1].toUInt() shl 8)
                or (this[2].toUInt() shl 16)
                or (this[3].toUInt() shl 24))
    else
        (this[3].toUInt()
                or (this[2].toUInt() shl 8)
                or (this[1].toUInt() shl 16)
                or (this[0].toUInt() shl 24))
}

/** Converts a List<UByte> (of size 8) to a ULong, taking endianness into account. */
fun List<UByte>.toULong(endian: Endian): ULong {
    require(size == 8) { "Expected 8 bytes for ULong, got $size" }
    return if (endian == Endian.LITTLE_ENDIAN) {
        (this[0].toULong()
                or (this[1].toULong() shl 8)
                or (this[2].toULong() shl 16)
                or (this[3].toULong() shl 24)
                or (this[4].toULong() shl 32)
                or (this[5].toULong() shl 40)
                or (this[6].toULong() shl 48)
                or (this[7].toULong() shl 56))
    } else {
        (this[7].toULong()
                or (this[6].toULong() shl 8)
                or (this[5].toULong() shl 16)
                or (this[4].toULong() shl 24)
                or (this[3].toULong() shl 32)
                or (this[2].toULong() shl 40)
                or (this[1].toULong() shl 48)
                or (this[0].toULong() shl 56))
    }
}

/** Converts a List<UByte> (of size 4) to an Int, taking endianness into account. */
fun List<UByte>.toInt(endian: Endian): Int = toUInt(endian).toInt()

/**
 * Multiplatform version of readDataByType.
 *
 * This function reads data of type [T] from the ByteArray starting at [offset].
 * It uses the provided [dataCount] and [endian] parameters to convert the raw bytes.
 */
inline fun <reified T> ByteArray.readDataByType(
    offset: Int,
    dataCount: Int = 1,
    endian: Endian = Endian.LITTLE_ENDIAN,
): List<T> {
    val bytesPerItem = getSizeInByte(T::class)
    val length = bytesPerItem * dataCount

    // Get the sub-array representing the bytes we want to read.
    val subArray = this.copyOfRange(offset, offset + length)
    // Convert each byte to UByte so we can perform unsigned arithmetic.
    val ubytes = subArray.map { it.toUByte() }

    return when (T::class) {
        UByte::class -> ubytes.map { it as T }
        Byte::class -> subArray.map { it as T }
        UShort::class -> ubytes.chunked(2)
            .map { chunk ->
                val value = if (endian == Endian.LITTLE_ENDIAN) {
                    (chunk[0].toUInt() or (chunk[1].toUInt() shl 8)).toUShort()
                } else {
                    (chunk[1].toUInt() or (chunk[0].toUInt() shl 8)).toUShort()
                }
                value as T
            }
        Short::class -> ubytes.chunked(2)
            .map { chunk ->
                val value = if (endian == Endian.LITTLE_ENDIAN) {
                    (chunk[0].toUInt() or (chunk[1].toUInt() shl 8)).toShort()
                } else {
                    (chunk[1].toUInt() or (chunk[0].toUInt() shl 8)).toShort()
                }
                value as T
            }

        UInt::class -> ubytes.chunked(4)
            .map { chunk: List<UByte> -> chunk.toUInt(endian) as T }

        ULong::class -> ubytes.chunked(8)
            .map { chunk -> chunk.toULong(endian) as T }

        Long::class -> ubytes.chunked(8)
            .map { chunk -> chunk.toULong(endian).toLong() as T }

        Int::class -> ubytes.chunked(4)
            .map { chunk -> chunk.toInt(endian) as T }

        Float::class -> ubytes.chunked(4)
            .map { chunk ->
                // Get the 4 bytes as UInt bits, then convert to Float.
                val intBits = chunk.toUInt(endian)
                Float.fromBits(intBits.toInt()) as T
            }
        Double::class -> ubytes.chunked(8)
            .map { chunk ->
                val longBits = chunk.toULong(endian).toLong()
                Double.fromBits(longBits) as T
            }

        Boolean::class -> ubytes.map { (it != 0.toUByte()) as T }

        else -> throw IllegalArgumentException("readDataByType: Unsupported type: ${T::class}")
    }
}

fun getSizeInByte(dataType: KClass<*>): Int {
    return when (dataType) {
        UByte::class -> 1
        Byte::class -> 1
        UShort::class -> 2
        Short::class -> 2
        UInt::class -> 4
        Int::class -> 4
        ULong::class -> 8
        Long::class -> 8
        Float::class -> 4
        Double::class -> 8
        Boolean::class -> 1
        else -> throw IllegalArgumentException("Unknown data type: $dataType")
    }
}

fun <E> List<E>.numberOfBytes(): Int {
    if (isEmpty()) {
        return 0
    }

    val elementSizeBytes = when (this[0]) {
        is UByte -> UByte.SIZE_BYTES
        is Byte -> Byte.SIZE_BYTES
        is UShort -> UShort.SIZE_BYTES
        is Short -> Short.SIZE_BYTES
        is UInt -> UInt.SIZE_BYTES
        is Int -> Int.SIZE_BYTES
        is Float -> Float.SIZE_BYTES
        is ULong -> ULong.SIZE_BYTES
        is Long -> Long.SIZE_BYTES
        is Double -> Double.SIZE_BYTES
        is Boolean -> 1 //TODO see if this is always true
        else -> throw IllegalArgumentException("Unsupported type: ${this[0]}")
    }
    return this.size * elementSizeBytes
}
