package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.tensor.ops.QuantizedMatmul
import sk.ainet.lang.tensor.ops.TernaryMatmul
import sk.ainet.lang.types.FP32

/**
 * `ops.matmul` with a packed right-hand operand gives what the deprecated scalar helpers
 * ([QuantizedMatmul], [TernaryMatmul]) compute from the same bytes (SKEEP-003, #1507). This is
 * what lets `matmulAutoDispatch` call `ops.matmul` instead of checking the weight's class.
 */
@Suppress("DEPRECATION")
class ScalarQuantHelperParityTest {

    private val ctx = DirectCpuExecutionContext()

    private fun input(batch: Int, inputDim: Int, seed: Int): Tensor<FP32, Float> {
        val rng = Random(seed)
        return ctx.fromFloatArray(Shape(batch, inputDim), FP32::class, FloatArray(batch * inputDim) { rng.nextFloat() * 2f - 1f })
    }

    @Suppress("UNCHECKED_CAST")
    private fun tensorOf(data: TensorData<*, *>): Tensor<FP32, Float> =
        ctx.fromData(data as TensorData<FP32, Float>, FP32::class)

    private fun assertSame(expected: Tensor<FP32, Float>, actual: Tensor<FP32, Float>) {
        assertEquals(expected.shape, actual.shape)
        val e = expected.data.copyToFloatArray()
        val a = actual.data.copyToFloatArray()
        for (i in e.indices) {
            val tol = 1e-3f * maxOf(1f, abs(e[i]))
            assertTrue(abs(e[i] - a[i]) <= tol, "output $i: helper ${e[i]}, ops.matmul ${a[i]}")
        }
    }

    @Test
    fun q8_0() {
        val inputDim = 64
        val outputDim = 5
        val bytes = ByteArray((inputDim / 32) * outputDim * 34).also { Random(1).nextBytes(it) }
        for (block in 0 until bytes.size / 34) { bytes[block * 34] = 0x00; bytes[block * 34 + 1] = 0x22 }
        val data = Q8_0BlockTensorData(Shape(inputDim, outputDim), bytes)
        val x = input(3, inputDim, 2)
        assertSame(QuantizedMatmul.matmulQ8_0(x, data, ctx), ctx.ops.matmul(x, tensorOf(data)))
        assertSame(QuantizedMatmul.matmulQ8_0(x, data, ctx), QuantizedMatmul.matmulAutoDispatch(x, tensorOf(data), ctx))
        assertTrue(QuantizedMatmul.isQ8_0Weight(tensorOf(data)))
        assertTrue(QuantizedMatmul.isQuantizedWeight(tensorOf(data)))
        assertTrue(!QuantizedMatmul.isQ4_KWeight(tensorOf(data)))
    }

    @Test
    fun q4_k() {
        val inputDim = 512
        val outputDim = 3
        val bytes = ByteArray((inputDim / 256) * outputDim * 144).also { Random(3).nextBytes(it) }
        for (block in 0 until bytes.size / 144) {
            val base = block * 144
            bytes[base] = 0x00; bytes[base + 1] = 0x22      // d, a small finite half
            bytes[base + 2] = 0x00; bytes[base + 3] = 0x1e  // dmin, a small finite half
        }
        val data = Q4_KBlockTensorData(Shape(inputDim, outputDim), bytes)
        val x = input(2, inputDim, 4)
        assertSame(QuantizedMatmul.matmulQ4_K(x, data, ctx), ctx.ops.matmul(x, tensorOf(data)))
        assertSame(QuantizedMatmul.matmulQ4_K(x, data, ctx), QuantizedMatmul.matmulAutoDispatch(x, tensorOf(data), ctx))
        assertTrue(QuantizedMatmul.isQ4_KWeight(tensorOf(data)))
        assertTrue(QuantizedMatmul.isQuantizedWeight(tensorOf(data)))
    }

    @Test
    fun ternary() {
        val inputDim = 24
        val outputDim = 4
        val rng = Random(5)
        val values = ByteArray(inputDim * outputDim) { (rng.nextInt(3) - 1).toByte() }
        val data = Ternary2BitTensorData.fromTernaryValues(Shape(inputDim, outputDim), values, scale = 0.5f)
        val x = input(2, inputDim, 6)
        assertSame(TernaryMatmul.matmul(x, data, ctx), ctx.ops.matmul(x, tensorOf(data)))
        assertSame(TernaryMatmul.matmul(x, data, ctx), TernaryMatmul.matmulAutoDispatch(x, tensorOf(data), ctx))
        assertTrue(TernaryMatmul.isTernaryWeight(tensorOf(data)))
        assertTrue(QuantizedMatmul.isQuantizedWeight(tensorOf(data)))
        assertTrue(!TernaryMatmul.isTernaryWeight(x))
    }
}
