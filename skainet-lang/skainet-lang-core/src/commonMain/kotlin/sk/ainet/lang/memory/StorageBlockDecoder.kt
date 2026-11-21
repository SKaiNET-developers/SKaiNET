package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.data.packedBlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Decodes blocks of a GGML block format (Q4_0, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0) out of the
 * [Storage] it is handed, whatever kind that is: heap, off-heap or a mapped file.
 *
 * It copies one block into a scratch array with [Storage.copyInto] and decodes it there. This is
 * the reference path of a view (`get`, `toFloatArray`, `materialize`); a kernel takes the storage
 * and never calls it.
 *
 * One instance holds one scratch block, so it must not decode from two threads at once.
 */
public class StorageBlockDecoder private constructor(
    override val blockSize: Int,
    override val bytesPerBlock: Int,
    private val scratch: ByteArray,
    private val oneBlock: PackedBlockStorage,
) : BlockDecoder {

    override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
        storage.copyInto(scratch, 0, blockIndex * bytesPerBlock, bytesPerBlock)
        oneBlock.dequantizeBlock(0, out, outOffset)
    }

    public companion object {
        /** A decoder for [encoding], or `null` when it is not one of the GGML block formats. */
        public fun orNull(encoding: TensorEncoding): StorageBlockDecoder? {
            val spec = encoding.blockSpec ?: return null
            if (spec.isPerTensor) return null
            val scratch = ByteArray(spec.bytesPerBlock)
            val oneBlock = packedBlockTensorData(encoding, Shape(spec.blockSize), scratch) as? PackedBlockStorage
                ?: return null
            return StorageBlockDecoder(spec.blockSize, spec.bytesPerBlock, scratch, oneBlock)
        }
    }
}
