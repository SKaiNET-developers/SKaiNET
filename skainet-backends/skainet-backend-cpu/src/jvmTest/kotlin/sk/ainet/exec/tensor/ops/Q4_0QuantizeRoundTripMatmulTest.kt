package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * End-to-end proof that the [Q4_0Quantizer] (FP32 → Q4_0) output is
 * directly consumable by the matmul dispatch — i.e. *any* loader that
 * produces dense FP32 weights can quantize them to Q4_0 and run
 * inference through the same kernel path GGUF Q4_0 weights use.
 *
 * Quantizes a dense weight, runs `ctx.ops.matmul(x, qWeight)`, and
 * checks it tracks the dense FP32 matmul within 4-bit error.
 */
class Q4_0QuantizeRoundTripMatmulTest {

    private val ctx = DirectCpuExecutionContext()

    @Suppress("UNCHECKED_CAST")
    private fun assertQuantizedTracksDense(inputDim: Int, outputDim: Int, seed: Int) {
        val rng = Random(seed)
        // Logical weight W[o][j] (output o, input j).
        val w = Array(outputDim) { FloatArray(inputDim) { rng.nextFloat() - 0.5f } }
        val inputV = FloatArray(inputDim) { rng.nextFloat() - 0.5f }

        // Reference: plain FP32 matmul.
        val expected = FloatArray(outputDim)
        for (o in 0 until outputDim) {
            var acc = 0f
            for (j in 0 until inputDim) acc += inputV[j] * w[o][j]
            expected[o] = acc
        }

        // Arrange weights in the kernel's packed block order — block
        // (blockIdx, o) holds the 32 input positions [blockIdx*32 .. +31]
        // for output o — then quantize that flat array. This is the layout
        // a loader producing Q4_0 matmul weights must emit.
        val blocks = inputDim / 32
        val flat = FloatArray(inputDim * outputDim)
        var p = 0
        for (blockIdx in 0 until blocks) {
            for (o in 0 until outputDim) {
                for (k in 0 until 32) {
                    flat[p++] = w[o][blockIdx * 32 + k]
                }
            }
        }
        val qData = Q4_0Quantizer.quantize(flat, Shape(inputDim, outputDim))
        val weight: Tensor<FP32, Float> = ctx.fromData(qData as TensorData<FP32, Float>, FP32::class)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, inputDim), FP32::class, inputV)

        val out = ctx.ops.matmul(input, weight).data.copyToFloatArray()

        // Q4_0 quantization error per weight is ~step/2 (step ≈ |max|/8 per
        // block); the dot-product error over `inputDim` random-signed terms
        // grows ~√blocks, not linearly. Tolerance scales accordingly.
        val tol = 0.1f + 0.1f * (inputDim / 32).coerceAtLeast(1)
        for (o in 0 until outputDim) {
            val diff = abs(expected[o] - out[o])
            assertTrue(
                diff <= tol,
                "quantized matmul drifted at $o: dense=${expected[o]} q4_0=${out[o]} diff=$diff tol=$tol",
            )
        }
    }

    @Test fun single_output_tracks_dense() =
        assertQuantizedTracksDense(inputDim = 64, outputDim = 1, seed = 1)

    @Test fun attention_proj_shape_tracks_dense() =
        assertQuantizedTracksDense(inputDim = 128, outputDim = 128, seed = 2)
}
