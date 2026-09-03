package sk.ainet.backend.api.kernel

/**
 * F32 input × Q4_K-packed weights matrix-vector multiply, in canonical
 * ggml super-block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Block layout (256-element super-block, 144 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q4_KTensorData] kdoc for the byte map):
 * - bytes 0..1  : `d` (super-block scale, FP16 LE)
 * - bytes 2..3  : `dMin` (super-block min-scale, FP16 LE)
 * - bytes 4..15 : 12 bytes of packed (6-bit scaleIdx, 6-bit minIdx) for
 *                 8 sub-blocks via ggml's `get_scale_min_k4` mixing
 * - bytes 16..143 : 128 bytes of 4-bit codes, *strided* in 4 groups of
 *                   32 bytes — each byte's lo nibble belongs to one
 *                   sub-block and the hi nibble of the same byte
 *                   belongs to the *next* sub-block over the same
 *                   intra-group index.
 *
 * Per sub-block s ∈ 0..7:
 *   `scale[s]  = d    * scaleIdx[s]`
 *   `offset[s] = dMin * minIdx[s]`
 *   per element: `dequant = code * scale[s] - offset[s]`
 *
 * The lazy-`dmin` accumulation trick (used by every well-tuned Q4_K
 * kernel including ggml's reference) avoids subtracting `offset` per
 * element by tracking `Σ(input · code)` and `Σ(input)` per sub-block
 * and combining as `scale * codeSum − offset * inputSum` once.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MAY assume
 * the arrays do not alias each other or `output`. They MUST fully
 * write the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight row-major contract: `weight` holds blocks laid out
 * `(blockIdx * outputDim + o) * 144` for output row `o` and input
 * block index `blockIdx`. This matches `Q4_KBlockTensorData.packedData`
 * and `JvmQuantizedVectorKernels.matmulQ4_KVec`.
 *
 * `inputDim` MUST be a multiple of 256 (the Q4_K block size).
 */
public interface Q4KMatmulKernel {
    /**
     * @param input FP32 input vector (single row).
     * @param inputOffset element offset into [input] where the row starts.
     * @param weight packed Q4_K bytes for the full `outputDim × inputDim` weight tensor.
     * @param weightByteOffset byte offset into [weight] where block (0, 0) starts.
     * @param inputDim contraction dimension (must be a multiple of 256).
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

    /**
     * Schedule-aware entry (SKEEP-005): a kernel that splits output rows across tasks takes the
     * split from [schedule]. The default ignores it and runs the legacy method, so an
     * implementation that has no parallel section is unaffected.
     */
    public fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
        schedule: sk.ainet.context.schedule.Schedule,
    ): Unit = matmul(input, inputOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
}
