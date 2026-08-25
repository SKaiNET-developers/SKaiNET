package sk.ainet.backend.api.kernel

/**
 * F32 input × Q5_0-packed weights matrix-vector multiply, in canonical
 * ggml block layout.
 *
 *   output[outputOffset + o] = Σ_j input[inputOffset + j] · dequant(weight[o, j])
 *     for j ∈ [0, inputDim), o ∈ [0, outputDim)
 *
 * Block layout (32-element block, 22 bytes/block; see
 * [sk.ainet.lang.tensor.data.Q5_0BlockTensorData] kdoc):
 * - bytes 0..1   : `d`  (block scale, FP16 LE)
 * - bytes 2..5   : `qh[0..3]` (the 5th/high bit of each of the 32 codes)
 * - bytes 6..21  : `qs[0..15]` (low 4 bits, two nibbles per byte)
 *
 * Per element, with `lo = qs[j] & 0x0F`, `hi = qs[j] >>> 4`, and the high
 * bits `bitLo = (qh[j/8] >>> (j%8)) & 1`, `bitHi = (qh[(j+16)/8] >>> ((j+16)%8)) & 1`:
 *
 *   element[j]      = d * (lo + (bitLo shl 4) - 16)   for j ∈ [0, 16)
 *   element[j + 16] = d * (hi + (bitHi shl 4) - 16)
 *
 * The `- 16` bias centres the unsigned 5-bit code around zero (no per-block
 * min). Matches `sk.ainet.io.gguf.dequant.DequantOps.dequantQ5_0FromBytes`.
 *
 * Implementations MUST NOT mutate `input` or `weight`. They MUST fully
 * write the `outputDim` floats starting at `output[outputOffset]`.
 *
 * Packed-weight **block-major** row contract: `weight` holds blocks laid
 * The weight is **input-block-major** (Q5_0BlockTensorData's bytes are canonical
 * row-major — a weight reaches this kernel through `TensorView.prepack`, not by
 * reinterpretation). One contract, written down in
 * `the Packed weight layout page in the docs site (explanation/packed-weight-layout)` (#973).
 *
 * `inputDim` MUST be a multiple of 32 (the Q5_0 block size).
 */
public interface Q5_0MatmulKernel {
    public fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )
}
