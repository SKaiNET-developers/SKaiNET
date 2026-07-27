package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarFp16MatmulKernel
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec

/**
 * Integration tests for the FP32 × FP16 dispatch path in `DefaultCpuOpsJvm.matmul`.
 *
 * The dispatch branch matches on `NarrowFloatTensorData` and then selects the kernel by the data's
 * `codec`. These tests prove an `Fp16DenseTensorData` weight reaches the FP16 kernel — and, more
 * importantly, that it is NOT decoded by the BF16 kernel, which would be silent garbage rather
 * than an error since both formats are 2 bytes wide.
 *
 * Mirrors [Bf16MatmulDispatchTest].
 */
class Fp16MatmulDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    /** binary16 carries 10 mantissa bits; accumulated error scales with `k`. */
    private val fp16TolPerK = 1.5e-3f

    private fun fp32ToFp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    private fun fp16Weight(inputDim: Int, outputDim: Int, seed: Int): Pair<Tensor<FP32, Float>, ByteArray> {
        val rng = Random(seed)
        val values = FloatArray(inputDim * outputDim) { rng.nextFloat() - 0.5f }
        val bytes = fp32ToFp16Bytes(values)
        val data = Fp16DenseTensorData(Shape(inputDim, outputDim), bytes) as TensorData<FP32, Float>
        return ctx.fromData(data, FP32::class) to bytes
    }

    private fun assertDispatchMatchesScalar(m: Int, k: Int, n: Int, seed: Int) {
        val rng = Random(seed)
        val inputFloats = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val (weight, weightBytes) = fp16Weight(k, n, seed)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, inputFloats)

        val outArr = ctx.ops.matmul(input, weight).data.copyToFloatArray()

        val expected = FloatArray(m * n)
        ScalarFp16MatmulKernel.matmul(inputFloats, 0, k, weightBytes, 0, n * 2, expected, 0, n, m, n, k)

        val tol = (fp16TolPerK * k.coerceAtLeast(1)).coerceAtLeast(fp16TolPerK)
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - outArr[i]) <= tol,
                "FP16 dispatch mismatch at $i: expected=${expected[i]} got=${outArr[i]} tol=$tol",
            )
        }
    }

    @Test
    fun single_batch_matmul_against_fp16_weight_routes_correctly() {
        assertDispatchMatchesScalar(m = 1, k = 128, n = 64, seed = 1)
    }

    @Test
    fun multi_batch_matmul_against_fp16_weight_routes_correctly() {
        assertDispatchMatchesScalar(m = 3, k = 256, n = 32, seed = 2)
    }

    @Test
    fun llm_typical_attention_proj_matmul_routes_correctly() {
        assertDispatchMatchesScalar(m = 1, k = 512, n = 512, seed = 3)
    }

    @Test
    fun fp16_weights_are_not_decoded_by_the_bf16_kernel() {
        // Both formats are 2 bytes wide, so a codec mix-up cannot fail loudly — it silently
        // produces wrong numbers. This pins that the dispatch picks by codec, not by width:
        // decoding the same bytes as BF16 must give a materially different answer, and the
        // dispatch must match the FP16 reading.
        val m = 1
        val k = 64
        val n = 8
        val rng = Random(42)
        val inputFloats = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val (weight, bytes) = fp16Weight(k, n, seed = 42)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, inputFloats)

        val dispatched = ctx.ops.matmul(input, weight).data.copyToFloatArray()

        val asFp16 = FloatArray(m * n)
        ScalarFp16MatmulKernel.matmul(inputFloats, 0, k, bytes, 0, n * 2, asFp16, 0, n, m, n, k)
        val asBf16 = FloatArray(m * n)
        ScalarBf16MatmulKernel.matmul(inputFloats, 0, k, bytes, 0, n * 2, asBf16, 0, n, m, n, k)

        val fp16Err = dispatched.indices.sumOf { abs(dispatched[it] - asFp16[it]).toDouble() }
        val bf16Err = dispatched.indices.sumOf { abs(dispatched[it] - asBf16[it]).toDouble() }

        assertTrue(fp16Err < 1e-3, "dispatch must match the FP16 decode, err=$fp16Err")
        assertTrue(
            bf16Err > fp16Err * 10,
            "the BF16 decode of the same bytes must differ materially " +
                "(fp16Err=$fp16Err bf16Err=$bf16Err); if these are close the test proves nothing",
        )
    }
}
