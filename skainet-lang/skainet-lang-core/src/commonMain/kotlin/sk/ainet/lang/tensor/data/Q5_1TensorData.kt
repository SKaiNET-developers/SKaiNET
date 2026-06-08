package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for the GGML **Q5_1** quantized format (5-bit, with a
 * per-block minimum).
 *
 * Q5_1 block format (32 elements per block, 24 bytes per block):
 * - 2 bytes: f16 scale (`d`)
 * - 2 bytes: f16 minimum (`m`)
 * - 4 bytes: `qh[0..3]` — the 5th (high) bit of each of the 32 codes
 * - 16 bytes: `qs[0..15]` — the low 4 bits, two nibbles per byte
 *
 * Dequantization (matching `sk.ainet.io.gguf.dequant.DequantOps.dequantQ5_1FromBytes`):
 * for `j ∈ [0, 16)`, with `q = qs[j]`, `lo = q & 0x0F`, `hi = q >>> 4`, and the
 * high bits `bitLo = (qh[j/8] >>> (j%8)) & 1`, `bitHi = (qh[(j+16)/8] >>> ((j+16)%8)) & 1`:
 *
 *   element[j]      = d * (lo + (bitLo shl 4)) + m
 *   element[j + 16] = d * (hi + (bitHi shl 4)) + m
 *
 * Enables direct quantized matmul without full dequantization, mirroring
 * [Q4_0TensorData] / [Q8_0TensorData].
 */
public interface Q5_1TensorData : TensorData<DType, Byte> {
    /** Number of Q5_1 blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    public companion object {
        /** Elements per Q5_1 block. */
        public const val BLOCK_SIZE: Int = 32

        /** Bytes per Q5_1 block (2 `d` + 2 `m` + 4 `qh` + 16 `qs`). */
        public const val BYTES_PER_BLOCK: Int = 24
    }
}

/**
 * Implementation of [Q5_1TensorData] backed by a packed byte array, in the
 * natural GGUF **row-major** `[out, in]` layout (each logical row's elements are
 * packed sequentially as `in / 32` blocks). `matmulQ5_1Vec` indexes the packed
 * bytes row-major, so no block-major re-layout is needed.
 */
public class Q5_1BlockTensorData(
    initialShape: Shape,
    private val data: ByteArray
) : Q5_1TensorData, PackedBlockStorage {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q5_1TensorData.BLOCK_SIZE - 1) / Q5_1TensorData.BLOCK_SIZE

    override val encoding: TensorEncoding get() = TensorEncoding.Q5_1
    override val blockSize: Int get() = Q5_1TensorData.BLOCK_SIZE

    init {
        val requiredBytes = blockCount * Q5_1TensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val base = blockIdx * Q5_1TensorData.BYTES_PER_BLOCK
        val d = Q4_0BlockTensorData.halfToFloat(((data[base + 1].toInt() and 0xFF) shl 8) or (data[base].toInt() and 0xFF))
        val m = Q4_0BlockTensorData.halfToFloat(((data[base + 3].toInt() and 0xFF) shl 8) or (data[base + 2].toInt() and 0xFF))
        val qh0 = data[base + 4].toInt() and 0xFF
        val qh1 = data[base + 5].toInt() and 0xFF
        val qh2 = data[base + 6].toInt() and 0xFF
        val qh3 = data[base + 7].toInt() and 0xFF
        val qh = intArrayOf(qh0, qh1, qh2, qh3)
        val qsBase = base + 8
        val elemsInBlock = minOf(Q5_1TensorData.BLOCK_SIZE, shape.volume - blockIdx * Q5_1TensorData.BLOCK_SIZE)
        for (j in 0 until 16) {
            val q = data[qsBase + j].toInt() and 0xFF
            val lo = q and 0x0F
            val hi = q ushr 4
            val bitLo = (qh[j / 8] ushr (j % 8)) and 0x01
            val bitHi = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 0x01
            val o0 = outputOffset + j
            if (j < elemsInBlock && o0 < output.size) output[o0] = d * (lo + (bitLo shl 4)) + m
            val o1 = outputOffset + 16 + j
            if (16 + j < elemsInBlock && o1 < output.size) output[o1] = d * (hi + (bitHi shl 4)) + m
        }
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val tmp = FloatArray(Q5_1TensorData.BLOCK_SIZE)
        val blockIdx = flatIndex / Q5_1TensorData.BLOCK_SIZE
        dequantizeBlock(blockIdx, tmp, 0)
        // Q5_1 stores real-valued reconstructions; expose the rounded code is not
        // meaningful, so this accessor is best-effort for debugging only.
        return tmp[flatIndex % Q5_1TensorData.BLOCK_SIZE].toInt().toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        throw UnsupportedOperationException("Q5_1BlockTensorData is read-only (packed quantized weights)")
    }

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }

    public companion object {
        /** Create [Q5_1BlockTensorData] from raw packed Q5_1 bytes (GGUF row-major). */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q5_1BlockTensorData =
            Q5_1BlockTensorData(shape, bytes)
    }
}

/** Dequantize Q5_1 tensor data to a FloatArray (row-major, matching the packed layout). */
public fun Q5_1TensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    val block = this as Q5_1BlockTensorData
    for (blockIdx in 0 until blockCount) {
        block.dequantizeBlock(blockIdx, result, blockIdx * Q5_1TensorData.BLOCK_SIZE)
    }
    return result
}
