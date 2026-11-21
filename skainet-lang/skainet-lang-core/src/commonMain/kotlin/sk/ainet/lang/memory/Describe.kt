package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Tensor

/**
 * One-line debugger rendering of a tensor (SKEEP-003 §4.7, PRD M0-F5):
 * `TensorId · Format · Shape · storage kind · origin · scope · StorageId`.
 * Fields that only exist from milestone M1 on (scope, storage id) print as `—`.
 *
 * Example: `model.layers.blk.3.attn.q_proj.weight · Float32/Q4_K · [2048, 2048] · Q4_KBlockTensorData · — · scope — · storage —`
 */
public fun Tensor<*, *>.describe(): String = buildString {
    append(id?.canonical ?: "—"); append(SEP)
    append(formatOrNull?.toString() ?: "?"); append(SEP)
    append(shapeText(data.shape.dimensions)); append(SEP)
    append(data::class.simpleName ?: "data"); append(SEP)
    append("—"); append(SEP)          // origin (file/offset) — storage-backed tensors only, M1
    append("scope —"); append(SEP)
    append("storage —")
}

/**
 * The same line for a view, with the fields a view knows: the kind of its storage and how it is
 * held (`Heap/Borrowed`, `Mapped/Owned`, ...), its scope and its storage id.
 *
 * Example: `model.blk.3.q_proj.weight · Float32/Q4_K · [2048, 2048] · Mapped/Owned · — · scope MODEL · storage #17`
 */
public fun TensorView.describe(): String = buildString {
    append(id?.canonical ?: "—"); append(SEP)
    append(format.toString()); append(SEP)
    append(shapeText(shape.dimensions)); append(SEP)
    val kind = when (storage) {
        is Storage.Heap -> "Heap"
        is Storage.OffHeap -> "OffHeap"
        is Storage.Mapped -> "Mapped"
        is Storage.Device -> "Device"
    }
    val held = when (storage.owner) {
        is Owner.Owned -> "Owned"
        is Owner.Borrowed -> "Borrowed"
        is Owner.Alias -> "Alias"
    }
    append(kind); append('/'); append(held); append(SEP)
    append("—"); append(SEP)          // origin (file/offset): the storage does not record it
    append("scope "); append(storage.scope.name); append(SEP)
    append("storage "); append(storage.id.toString())
}

private const val SEP = " · "

private fun shapeText(dims: IntArray): String = dims.joinToString(", ", "[", "]")
