package sk.ainet.backend.api.kernel

/**
 * F32 input × Q5_K-packed weights matrix-vector multiply, in canonical
 * ggml super-block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Block layout (256-element super-block, 176 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q5_KTensorData] kdoc for the byte map):
 * - bytes 0..1   : `d` (super-block scale, FP16 LE)
 * - bytes 2..3   : `dMin` (super-block min-scale, FP16 LE)
 * - bytes 4..15  : 12 bytes of packed (6-bit scaleIdx, 6-bit minIdx) for
 *                  8 sub-blocks via ggml's `get_scale_min_k4` mixing
 *                  (identical to Q4_K)
 * - bytes 16..47 : 32 bytes `qh` high-bit plane (the 5th bit of each code)
 * - bytes 48..175: 128 bytes of 4-bit low nibbles, *strided* in 4 groups of
 *                  32 bytes (identical layout to Q4_K's `qs`)
 *
 * Per sub-block s ∈ 0..7:
 *   `scale[s]  = d    * scaleIdx[s]`
 *   `offset[s] = dMin * minIdx[s]`
 *   per element: `code = lowNibble | (fifthBit << 4)` (0..31);
 *                `dequant = code * scale[s] - offset[s]`
 *
 * The lazy-`dmin` accumulation trick (used by every well-tuned K-quant
 * kernel including ggml's reference) avoids subtracting `offset` per
 * element by tracking `Σ(input · code)` and `Σ(input)` per sub-block
 * and combining as `scale * codeSum − offset * inputSum` once.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MAY assume
 * the arrays do not alias each other or `output`. They MUST fully
 * write the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight row-major contract: `weight` holds blocks laid out
 * `(blockIdx * outputDim + o) * 176` for output row `o` and input
 * block index `blockIdx`. This matches `Q5_KBlockTensorData.packedData`.
 *
 * `inputDim` MUST be a multiple of 256 (the Q5_K block size).
 */
public interface Q5KMatmulKernel {
    /**
     * @param input FP32 input vector (single row).
     * @param inputOffset element offset into [input] where the row starts.
     * @param weight packed Q5_K bytes for the full `outputDim × inputDim` weight tensor.
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
}
