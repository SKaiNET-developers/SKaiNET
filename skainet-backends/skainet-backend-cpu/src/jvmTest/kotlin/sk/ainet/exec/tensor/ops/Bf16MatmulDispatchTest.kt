package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * Integration tests for the FP32 × BF16 dispatch path in
 * [DefaultCpuOpsJvm.matmul]. Confirms `ops.matmul` against a
 * `Bf16DenseTensorData` weight produces the same output as the
 * `ScalarBf16MatmulKernel` reference — proving the new `is
 * Bf16TensorData ->` branch in `chooseQuantizedMatmul` is reached
 * and routes through the BF16 SPI (or its scalar fallback when no
 * SIMD provider resolves). Mirrors `Q8_0MatmulDispatchTest` in
 * shape and coverage.
 */
class Bf16MatmulDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    /** BF16 has 7 mantissa bits — accumulated error scales with `k`. */
    private val bf16TolPerK = 1e-2f

    /** Truncate FP32 → BF16 (high 16 bits, zero rounding), pack LE bytes. */
    private fun fp32ToBf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bf16 = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bf16 and 0xFF).toByte()
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte()
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    private fun bf16Weight(inputDim: Int, outputDim: Int, seed: Int): Pair<Tensor<FP32, Float>, ByteArray> {
        val rng = Random(seed)
        val values = FloatArray(inputDim * outputDim) { rng.nextFloat() - 0.5f }
        val bytes = fp32ToBf16Bytes(values)
        val data = Bf16DenseTensorData(Shape(inputDim, outputDim), bytes) as TensorData<FP32, Float>
        return ctx.fromData(data, FP32::class) to bytes
    }

    private fun scalarReference(
        input: FloatArray, weightBytes: ByteArray,
        m: Int, n: Int, k: Int,
    ): FloatArray {
        val out = FloatArray(m * n)
        ScalarBf16MatmulKernel.matmul(
            input, 0, k,
            weightBytes, 0, n * 2,
            out, 0, n,
            m, n, k,
        )
        return out
    }

    private fun assertDispatchMatchesScalar(
        m: Int, k: Int, n: Int, seed: Int,
    ) {
        val rng = Random(seed)
        val inputFloats = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val (weight, weightBytes) = bf16Weight(k, n, seed)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, inputFloats)

        val out = ctx.ops.matmul(input, weight)
        val outArr = out.data.copyToFloatArray()
        val expected = scalarReference(inputFloats, weightBytes, m, n, k)

        val tol = (bf16TolPerK * k.coerceAtLeast(1)).coerceAtLeast(bf16TolPerK)
        for (i in expected.indices) {
            val diff = abs(expected[i] - outArr[i])
            assertTrue(
                diff <= tol,
                "BF16 dispatch mismatch at $i: expected=${expected[i]} got=${outArr[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test
    fun single_batch_matmul_against_bf16_weight_routes_correctly() {
        assertDispatchMatchesScalar(m = 1, k = 128, n = 64, seed = 1)
    }

    @Test
    fun multi_batch_matmul_against_bf16_weight_routes_correctly() {
        assertDispatchMatchesScalar(m = 3, k = 256, n = 32, seed = 2)
    }

    @Test
    fun llm_typical_attention_proj_matmul_routes_correctly() {
        assertDispatchMatchesScalar(m = 1, k = 512, n = 512, seed = 3)
    }
}
