package sk.ainet.data.common

/**
 * Raised when a built-in dataset loader is compiled for a target where the
 * required transport, archive, or decompression primitive is not implemented.
 */
public class DatasetLoaderUnsupportedTargetException(
    public val dataset: String,
    public val target: String,
    reason: String
) : UnsupportedOperationException("$dataset loader is not supported on $target: $reason")

/** Throws a typed unsupported-target exception for built-in dataset loaders. */
public fun unsupportedDatasetLoader(dataset: String, target: String, reason: String): Nothing {
    throw DatasetLoaderUnsupportedTargetException(dataset, target, reason)
}

/** Returns true when this byte array starts with the gzip magic header. */
public fun ByteArray.hasGzipHeader(): Boolean =
    size >= 2 && this[0] == 0x1f.toByte() && this[1] == 0x8b.toByte()
