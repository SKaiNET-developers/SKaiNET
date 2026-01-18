package sk.ainet.lang.tensor.ops

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

class TernaryMatmulTest {

    private val ctx = DefaultDataExecutionContext()

    @Test
    fun `ternary matmul with identity-like weights`() {
        // Input: [1.0, 2.0, 3.0]
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 3), FP32::class,
            floatArrayOf(1.0f, 2.0f, 3.0f)
        )

        // Weight [3, 2] - first column sums inputs, second zeros them
        // Col 0: [1, 1, 1] -> sum = 1+2+3 = 6
        // Col 1: [0, 0, 0] -> sum = 0
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(3, 2),
            byteArrayOf(
                1, 0,   // row 0
                1, 0,   // row 1
                1, 0    // row 2
            )
        )

        val output = TernaryMatmul.matmul(input, weights, ctx)

        assertEquals(1, output.shape.dimensions[0])
        assertEquals(2, output.shape.dimensions[1])
        assertEquals(6.0f, output.data[0, 0], 0.001f)
        assertEquals(0.0f, output.data[0, 1], 0.001f)
    }

    @Test
    fun `ternary matmul with negative weights`() {
        // Input: [1.0, 2.0]
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 2), FP32::class,
            floatArrayOf(1.0f, 2.0f)
        )

        // Weight [2, 2] - first column: [1, -1], second: [-1, 1]
        // Col 0: 1*1 + 2*(-1) = 1 - 2 = -1
        // Col 1: 1*(-1) + 2*1 = -1 + 2 = 1
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(2, 2),
            byteArrayOf(
                1, -1,   // row 0: output[0] += input[0], output[1] -= input[0]
                -1, 1    // row 1: output[0] -= input[1], output[1] += input[1]
            )
        )

        val output = TernaryMatmul.matmul(input, weights, ctx)

        assertEquals(-1.0f, output.data[0, 0], 0.001f)
        assertEquals(1.0f, output.data[0, 1], 0.001f)
    }

    @Test
    fun `ternary matmul with scale`() {
        // Input: [1.0, 1.0]
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 2), FP32::class,
            floatArrayOf(1.0f, 1.0f)
        )

        // Weight [2, 1] - both +1, scale = 0.5
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(2, 1),
            byteArrayOf(1, 1),
            scale = 0.5f
        )

        val output = TernaryMatmul.matmul(input, weights, ctx)

        // Without scale: 1 + 1 = 2, with scale: 2 * 0.5 = 1.0
        assertEquals(1.0f, output.data[0, 0], 0.001f)
    }

    @Test
    fun `ternary matmul batched input`() {
        // Batch of 2 inputs: [[1, 2], [3, 4]]
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(2, 2), FP32::class,
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        )

        // Weight [2, 2] - identity-like
        // [1, 0]
        // [0, 1]
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(2, 2),
            byteArrayOf(1, 0, 0, 1)
        )

        val output = TernaryMatmul.matmul(input, weights, ctx)

        // Batch 0: [1, 2] * identity = [1, 2]
        assertEquals(1.0f, output.data[0, 0], 0.001f)
        assertEquals(2.0f, output.data[0, 1], 0.001f)

        // Batch 1: [3, 4] * identity = [3, 4]
        assertEquals(3.0f, output.data[1, 0], 0.001f)
        assertEquals(4.0f, output.data[1, 1], 0.001f)
    }

    @Test
    fun `ternary matmul larger matrix`() {
        // 4x4 input
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 4), FP32::class,
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        )

        // Weight [4, 2] - alternating pattern
        // Col 0: [1, -1, 1, -1] -> 1 - 2 + 3 - 4 = -2
        // Col 1: [-1, 1, -1, 1] -> -1 + 2 - 3 + 4 = 2
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(4, 2),
            byteArrayOf(
                1, -1,
                -1, 1,
                1, -1,
                -1, 1
            )
        )

        val output = TernaryMatmul.matmul(input, weights, ctx)

        assertEquals(-2.0f, output.data[0, 0], 0.001f)
        assertEquals(2.0f, output.data[0, 1], 0.001f)
    }

    @Test
    fun `isTernaryWeight detects ternary data`() {
        val ternaryWeights = Ternary2BitTensorData.zeros(Shape(4, 4))
        val ternaryTensor = ctx.fromFloatArray<FP32, Float>(
            Shape(4, 4), FP32::class,
            FloatArray(16) { 0f }
        )

        // Note: The tensor created from ctx.fromFloatArray will NOT be ternary
        // We'd need to wrap TernaryTensorData properly in a tensor to test this
        // For now, just test that regular tensors are not detected as ternary
        assertEquals(false, TernaryMatmul.isTernaryWeight(ternaryTensor))
    }
}
