package sk.ainet.backend.api.kernel

/**
 * F32 input × Q5_1-packed weights matrix-vector multiply, in canonical
 * ggml block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Block layout (32-element block, 24 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q5_1BlockTensorData] kdoc):
 * - bytes 0..1   : `d`  (block scale, FP16 LE)
 * - bytes 2..3   : `m`  (block minimum, FP16 LE)
 * - bytes 4..7   : `qh[0..3]` (the 5th/high bit of each of the 32 codes)
 * - bytes 8..23  : `qs[0..15]` (low 4 bits, two nibbles per byte)
 *
 * Per element, with `lo = qs[j] & 0x0F`, `hi = qs[j] >>> 4`, and the high
 * bits `bitLo = (qh[j/8] >>> (j%8)) & 1`, `bitHi = (qh[(j+16)/8] >>> ((j+16)%8)) & 1`:
 *
 *   element[j]      = d * (lo + (bitLo shl 4)) + m   for j ∈ [0, 16)
 *   element[j + 16] = d * (hi + (bitHi shl 4)) + m
 *
 * Matches `sk.ainet.io.gguf.dequant.DequantOps.dequantQ5_1FromBytes`.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MAY assume
 * the arrays do not alias each other or `output`. They MUST fully write
 * the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight **block-major** row contract: `weight` holds blocks laid
 * out `(blockIdx * outputDim + o) * 24` for output row `o` and input
 * block index `blockIdx`. This matches `Q5_1BlockTensorData.packedData`
 * after the GGUF row-major → input-block-major re-layout.
 *
 * `inputDim` MUST be a multiple of 32 (the Q5_1 block size).
 */
public interface Q5_1MatmulKernel {
    /**
     * @param input FP32 input vector (single row).
     * @param inputOffset element offset into [input] where the row starts.
     * @param weight packed Q5_1 bytes for the full `outputDim × inputDim` weight tensor.
     * @param weightByteOffset byte offset into [weight] where block (0, 0) starts.
     * @param inputDim contraction dimension (must be a multiple of 32).
     * @param outputDim number of output cells.
     * @param output FP32 output vector.
     * @param outputOffset element offset into [output] where the row starts.
     */
    public fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )
}
