package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarQ8_0MatmulKernel
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * Integration tests for the FP32 × Q8_0 dispatch path in
 * [DefaultCpuOpsJvm.matmul]. Confirms that calling matmul on a
 * Q8_0-backed weight tensor produces the same output as the scalar
 * Q8_0 kernel — proving the dispatch actually routes through the
 * registered Q8_0 SPI kernel (or its legacy `JvmQuantizedVectorKernels`
 * fallback when the SPI doesn't resolve). Either path is correct;
 * this test pins integration, not kernel correctness (already covered
 * by the per-kernel parity tests in #606).
 */
class Q8_0MatmulDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    private val blockSize = 32
    private val bytesPerBlock = 34

    private fun randomQ8_0Bytes(blocksPerInputDim: Int, outputDim: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val numBlocks = blocksPerInputDim * outputDim
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // FP16 scale ≈ 7.6e-3 (low-bit FP16 0x2200) — safely finite, non-zero.
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x22.toByte()
        }
        return bytes
    }

    private fun ScalarQ8_0_reference(
        input: FloatArray, weight: ByteArray,
        inputDim: Int, outputDim: Int,
        batchSize: Int,
    ): FloatArray {
        val out = FloatArray(batchSize * outputDim)
        for (b in 0 until batchSize) {
            ScalarQ8_0MatmulKernel.matmul(
                input, b * inputDim,
                weight, 0,
                inputDim, outputDim,
                out, b * outputDim,
            )
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    private fun q8_0Tensor(inputDim: Int, outputDim: Int, seed: Int): Tensor<FP32, Float> {
        val blocksPerInputDim = inputDim / blockSize
        val bytes = randomQ8_0Bytes(blocksPerInputDim, outputDim, seed)
        // Logical shape of a Q8_0 weight tensor is [inputDim, outputDim].
        val data = Q8_0BlockTensorData(Shape(inputDim, outputDim), bytes)
        return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
    }

    private fun assertDispatchMatchesScalar(
        batchSize: Int, inputDim: Int, outputDim: Int, seed: Int,
        tolPerBlock: Float = 1e-2f,
    ) {
        val rng = Random(seed)
        val inputFloats = FloatArray(batchSize * inputDim) { rng.nextFloat() - 0.5f }
        val blocksPerInputDim = inputDim / blockSize

        val weightBytes = randomQ8_0Bytes(blocksPerInputDim, outputDim, seed)
        val weight = q8_0Tensor(inputDim, outputDim, seed).let { t ->
            // q8_0Tensor regenerates bytes from seed → use the SAME byte buffer
            // for the scalar reference path so the comparison is honest.
            @Suppress("UNCHECKED_CAST")
            val td = Q8_0BlockTensorData(Shape(inputDim, outputDim), weightBytes) as TensorData<FP32, Float>
            ctx.fromData(td, FP32::class)
        }
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(batchSize, inputDim), FP32::class, inputFloats,
        )

        val out = ctx.ops.matmul(input, weight)
        val outArr = out.data.copyToFloatArray()

        val expected = ScalarQ8_0_reference(inputFloats, weightBytes, inputDim, outputDim, batchSize)

        val tol = (tolPerBlock * blocksPerInputDim.coerceAtLeast(1)).coerceAtLeast(tolPerBlock)
        for (i in expected.indices) {
            val diff = abs(expected[i] - outArr[i])
            assertTrue(
                diff <= tol,
                "dispatch mismatch at $i: expected=${expected[i]} got=${outArr[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test
    fun single_batch_matmul_against_q8_0_weight_routes_correctly() {
        // batchSize=1 hits the optimized "no copyOfRange" branch in chooseQuantizedMatmul.
        assertDispatchMatchesScalar(batchSize = 1, inputDim = 128, outputDim = 64, seed = 1)
    }

    @Test
    fun multi_batch_matmul_against_q8_0_weight_routes_correctly() {
        // batchSize>1 exercises the per-row copyOfRange branch.
        assertDispatchMatchesScalar(batchSize = 3, inputDim = 256, outputDim = 32, seed = 2)
    }

    @Test
    fun llm_typical_attention_proj_matmul_routes_correctly() {
        // Realistic attention-projection size (matvec at dim×dim).
        assertDispatchMatchesScalar(batchSize = 1, inputDim = 512, outputDim = 512, seed = 3)
    }
}
