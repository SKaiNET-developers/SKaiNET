@file:JvmName("TensorSpecs")

package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.jvm.JvmName

/**
 * Metadata key used to carry a [TensorEncoding] on a [TensorSpec].
 *
 * Exposed so that callers that need to read/write the raw metadata map
 * directly (for interop, serialization round-trips, etc.) use the same
 * string the typed accessors below use.
 */
public const val TENSOR_ENCODING_METADATA_KEY: String = "tensorEncoding"

/**
 * Physical storage encoding carried on this spec, or `null` if the producer
 * did not populate it.
 *
 * A `null` return means "unknown / not carried through the graph" — it is
 * NOT equivalent to [TensorEncoding.Dense]. Consumers that need a concrete
 * encoding should treat `null` as unknown and fall back to dtype-driven
 * defaults rather than assuming dense.
 */
public val TensorSpec.tensorEncoding: TensorEncoding?
    get() = metadata[TENSOR_ENCODING_METADATA_KEY] as? TensorEncoding

/**
 * Return a copy of this spec with [encoding] stored in its metadata map.
 * Passing `null` removes the entry; passing a non-null value adds or
 * replaces it, leaving all other metadata untouched.
 */
public fun TensorSpec.withTensorEncoding(encoding: TensorEncoding?): TensorSpec {
    val newMetadata: Map<String, Any> = if (encoding == null) {
        metadata - TENSOR_ENCODING_METADATA_KEY
    } else {
        metadata + (TENSOR_ENCODING_METADATA_KEY to encoding)
    }
    return copy(metadata = newMetadata)
}

/**
 * Infer a [TensorEncoding] from a concrete [TensorData] instance, or return
 * `null` when the layout is dense / unknown. Single source of truth for the
 * data-subclass → encoding mapping so trace builders and loaders agree.
 *
 * Any [TensorData] implementing [PackedBlockStorage] already exposes its
 * own `encoding`, so this helper is one line today but centralizes the
 * contract for future non-packed quantized layouts.
 */
public fun TensorData<*, *>.inferTensorEncoding(): TensorEncoding? = when (this) {
    is PackedBlockStorage -> this.encoding
    else -> null
}
