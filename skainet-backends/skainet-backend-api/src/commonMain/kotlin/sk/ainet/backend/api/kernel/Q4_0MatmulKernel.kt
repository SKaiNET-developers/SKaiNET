package sk.ainet.backend.api.kernel

/**
 * F32 input × Q4_0-packed weights matrix-vector multiply, in canonical
 * ggml block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Block layout (32-element block, 18 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q4_0BlockTensorData] kdoc):
 * - bytes 0..1  : `d` (block scale, FP16 LE)
 * - bytes 2..17 : 16 bytes packing 32 4-bit codes (split layout — low
 *   nibbles decode elements 0..15, high nibbles decode elements 16..31)
 *
 * Per element: `dequant = (code - 8) * d` (the `- 8` bias centres the
 * unsigned 4-bit code around zero). Q4_0 has no per-block min / offset.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MAY assume
 * the arrays do not alias each other or `output`. They MUST fully
 * write the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight row-major contract: `weight` holds blocks laid out
 * `(blockIdx * outputDim + o) * 18` for output row `o` and input block
 * index `blockIdx`. This matches `Q4_0BlockTensorData.packedData`.
 *
 * `inputDim` MUST be a multiple of 32 (the Q4_0 block size).
 */
public interface Q4_0MatmulKernel {
    /**
     * @param input FP32 input vector (single row).
     * @param inputOffset element offset into [input] where the row starts.
     * @param weight packed Q4_0 bytes for the full `outputDim × inputDim` weight tensor.
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
