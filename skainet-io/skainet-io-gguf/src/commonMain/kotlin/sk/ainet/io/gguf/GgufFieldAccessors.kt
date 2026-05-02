package sk.ainet.io.gguf

/**
 * Typed accessors for the raw `Map<String, Any?>` returned by `StreamingGGUFReader.fields`
 * (and any other producer of GGUF metadata).
 *
 * GGUF stores most numeric fields as `uint32` / `uint64`, which the reader preserves as
 * Kotlin `UInt` / `ULong`. Those unsigned types do **not** extend `kotlin.Number`, so the
 * naïve idiom `(value as? Number)?.toInt()` silently returns `null` for any modern GGUF.
 * These accessors handle every signed and unsigned integer type the reader can emit, plus
 * the string-encoded numeric variant some metadata fields use.
 *
 * All accessors take a vararg of keys and return the first match — useful for fields that
 * have moved namespaces between architectures (`llama.context_length` vs.
 * `general.context_length` vs. `model.context_length`).
 */

public fun Map<String, Any?>.getString(vararg keys: String): String? {
    for (key in keys) {
        val value = this[key]
        if (value is String) return value
    }
    return null
}

public fun Map<String, Any?>.getInt(vararg keys: String): Int? {
    for (key in keys) {
        when (val value = this[key]) {
            is Int -> return value
            is UInt -> return value.toInt()
            is Long -> return value.toInt()
            is ULong -> return value.toInt()
            is Short -> return value.toInt()
            is UShort -> return value.toInt()
            is Byte -> return value.toInt()
            is UByte -> return value.toInt()
            is Number -> return value.toInt()
            is String -> value.toIntOrNull()?.let { return it }
        }
    }
    return null
}

public fun Map<String, Any?>.getLong(vararg keys: String): Long? {
    for (key in keys) {
        when (val value = this[key]) {
            is Long -> return value
            is ULong -> return value.toLong()
            is Int -> return value.toLong()
            is UInt -> return value.toLong()
            is Short -> return value.toLong()
            is UShort -> return value.toLong()
            is Byte -> return value.toLong()
            is UByte -> return value.toLong()
            is Number -> return value.toLong()
            is String -> value.toLongOrNull()?.let { return it }
        }
    }
    return null
}

public fun Map<String, Any?>.getIntList(vararg keys: String): List<Int>? {
    for (key in keys) {
        val value = this[key] ?: continue
        val ints = when (value) {
            is List<*> -> value.mapNotNull { it.numberToIntOrNull() }
            is Array<*> -> value.mapNotNull { it.numberToIntOrNull() }
            is IntArray -> value.toList()
            is LongArray -> value.map { it.toInt() }
            is ShortArray -> value.map { it.toInt() }
            is ByteArray -> value.map { it.toInt() }
            is UIntArray -> value.map { it.toInt() }
            is ULongArray -> value.map { it.toInt() }
            is UShortArray -> value.map { it.toInt() }
            is UByteArray -> value.map { it.toInt() }
            else -> null
        }
        if (ints != null && ints.isNotEmpty()) return ints
    }
    return null
}

public fun Map<String, Any?>.getStringList(vararg keys: String): List<String>? {
    for (key in keys) {
        val value = this[key]
        when (value) {
            is List<*> -> value.filterIsInstance<String>().takeIf { it.isNotEmpty() }
                ?.let { return it }
            is Array<*> -> value.filterIsInstance<String>().takeIf { it.isNotEmpty() }
                ?.let { return it }
        }
    }
    return null
}

private fun Any?.numberToIntOrNull(): Int? = when (this) {
    is Int -> this
    is UInt -> this.toInt()
    is Long -> this.toInt()
    is ULong -> this.toInt()
    is Short -> this.toInt()
    is UShort -> this.toInt()
    is Byte -> this.toInt()
    is UByte -> this.toInt()
    is Number -> this.toInt()
    else -> null
}
