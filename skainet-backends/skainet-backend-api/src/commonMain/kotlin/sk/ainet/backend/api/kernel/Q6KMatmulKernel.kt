package sk.ainet.backend.api.kernel

/**
 * F32 input × Q6_K-packed weights matrix-vector multiply, in canonical
 * ggml block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Q6_K super-block layout (256 elements, 210 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q6_KBlockTensorData]):
 * - bytes 0..127   : `ql[0..127]`   (lower 4 bits of each code)
 * - bytes 128..191 : `qh[0..63]`    (upper 2 bits of each code)
 * - bytes 192..207 : `scales[0..15]`(int8 per-16-element sub-block scales)
 * - bytes 208..209 : `d`            (super-block scale, FP16 LE)
 *
 * The 6-bit signed code is reassembled from `ql`/`qh` (see ggml
 * `dequantize_row_q6_K`); per element `dequant = d * scales[sub] * (code - 32)`.
 * Matches `sk.ainet.io.gguf.dequant.DequantOps.dequantQ6KFromBytes` — that is
 * the authoritative reference; implementations MUST agree with it.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MUST fully
 * write the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight **block-major** row contract: blocks laid out
 * The weight is **input-block-major** (Q6_KBlockTensorData's bytes are canonical
 * row-major — a weight reaches this kernel through `TensorView.prepack`, not by
 * reinterpretation). One contract, written down in
 * `the Packed weight layout page in the docs site (explanation/packed-weight-layout)` (#973).
 *
 * `inputDim` MUST be a multiple of 256 (the Q6_K super-block size).
 */
public interface Q6KMatmulKernel {
    public fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )
}
