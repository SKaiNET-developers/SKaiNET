package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * The heap block tensor data for [encoding] over [bytes], or `null` for an encoding that has no
 * heap block class (the ternary and dense encodings).
 *
 * One place that maps an encoding to its data class, so code outside this package can build or
 * relabel a packed weight from its `Format` without naming the seven classes (SKEEP-003, #1507).
 *
 * @param blockOrder the order [bytes]' blocks are in
 * @param blockAxis the axis of [shape] the blocks run along; see
 *   [sk.ainet.lang.tensor.storage.PackedBlockStorage.blockAxis]
 */
public fun packedBlockTensorData(
    encoding: TensorEncoding,
    shape: Shape,
    bytes: ByteArray,
    blockOrder: BlockOrder = BlockOrder.ROW_MAJOR,
    blockAxis: Int = shape.rank - 1,
): TensorData<DType, Byte>? = when (encoding) {
    TensorEncoding.Q4_K -> Q4_KBlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q5_K -> Q5_KBlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q6_K -> Q6_KBlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q5_1 -> Q5_1BlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q5_0 -> Q5_0BlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q8_0 -> Q8_0BlockTensorData(shape, bytes, blockOrder, blockAxis)
    TensorEncoding.Q4_0 -> Q4_0BlockTensorData(shape, bytes, blockOrder, blockAxis)
    else -> null
}
