@file:JvmName("TensorSpecIdentities")

package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.TensorId
import kotlin.jvm.JvmName

/**
 * Metadata key used to carry a [TensorId] on a [TensorSpec] (#1178).
 *
 * Same contract as [TENSOR_ENCODING_METADATA_KEY]: an untyped metadata entry with typed
 * accessors, so the compile pipeline carries the fact without importing storage-model types —
 * the `TensorSpecEncoding` precedent, and the standing rule for everything the tape captures.
 */
public const val TENSOR_ID_METADATA_KEY: String = "tensorId"

/**
 * The module-path identity carried on this spec (`model.layers[3].attn.q_proj.weight`), or
 * `null` if no producer knew it. `null` means "unidentified", not "anonymous by design" —
 * consumers keying per-weight policy on identity should skip unidentified tensors.
 */
public val TensorSpec.tensorId: TensorId?
    get() = metadata[TENSOR_ID_METADATA_KEY] as? TensorId

/**
 * Return a copy of this spec with [id] stored in its metadata map. Passing `null` removes the
 * entry; a non-null value adds or replaces it, leaving all other metadata untouched.
 */
public fun TensorSpec.withTensorId(id: TensorId?): TensorSpec {
    val newMetadata: Map<String, Any> = if (id == null) {
        metadata - TENSOR_ID_METADATA_KEY
    } else {
        metadata + (TENSOR_ID_METADATA_KEY to id)
    }
    return copy(metadata = newMetadata)
}

/**
 * Metadata key for the physical block order of packed storage, carried as a **stable string**
 * (`ROW_MAJOR` / `INPUT_BLOCK_MAJOR`) rather than the `sk.ainet.lang.memory.BlockOrder` enum —
 * deliberately, so the compile pipeline reads it without importing storage-model types (#1179).
 */
public const val TENSOR_BLOCK_ORDER_METADATA_KEY: String = "tensorBlockOrder"

/** The packed block order carried on this spec, or `null` when unknown / not packed. */
public val TensorSpec.blockOrder: String?
    get() = metadata[TENSOR_BLOCK_ORDER_METADATA_KEY] as? String

/** Return a copy of this spec with [order] stored in its metadata map (`null` removes the entry). */
public fun TensorSpec.withBlockOrder(order: String?): TensorSpec {
    val newMetadata: Map<String, Any> = if (order == null) {
        metadata - TENSOR_BLOCK_ORDER_METADATA_KEY
    } else {
        metadata + (TENSOR_BLOCK_ORDER_METADATA_KEY to order)
    }
    return copy(metadata = newMetadata)
}
