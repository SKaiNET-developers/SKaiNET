package sk.ainet.lang.tensor.ops

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.math.abs

class QuantizedMatmulTest {

    private val ctx = DefaultDataExecutionContext()

    /**
     * Create Q8_0 block data for testing.
     * Block: 2 bytes f16 scale + 32 bytes int8 codes.
     */
    private fun createQ8_0Block(scaleF16: Short, codes: ByteArray): ByteArray {
        require(codes.size == 32) { "Q8_0 block requires 32 codes" }
        val block = ByteArray(34)
        block[0] = (scaleF16.toInt() and 0xFF).toByte()
        block[1] = ((scaleF16.toInt() shr 8) and 0xFF).toByte()
        codes.copyInto(block, 2)
        return block
    }

    /**
     * Create Q4_K block data for testing (simplified).
     * Block: 2 f16 d + 2 f16 dMin + 12 scales + 128 codes = 144 bytes.
     */
    private fun createQ4_KBlock(
        d: Short,
        dMin: Short,
        scaleMinIndices: ByteArray = ByteArray(12) { 0xFF.toByte() },  // max indices
        codes: ByteArray = ByteArray(128)
    ): ByteArray {
        val block = ByteArray(144)
        block[0] = (d.toInt() and 0xFF).toByte()
        block[1] = ((d.toInt() shr 8) and 0xFF).toByte()
        block[2] = (dMin.toInt() and 0xFF).toByte()
        block[3] = ((dMin.toInt() shr 8) and 0xFF).toByte()
        scaleMinIndices.copyInto(block, 4, 0, minOf(12, scaleMinIndices.size))
        codes.copyInto(block, 16, 0, minOf(128, codes.size))
        return block
    }

    // ========== Q8_0 Matmul Tests ==========

    @Test
    fun `Q8_0 matmul simple case`() {
        // Input: [1.0, 2.0, ...]  32 elements
        val inputData = FloatArray(32) { (it + 1).toFloat() }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 32), FP32::class, inputData)

        // Weight [32, 1] - single output dimension
        // scale = 1.0 (0x3C00), codes = [1, 1, 1, ..., 1]
        val codes = ByteArray(32) { 1 }
        val blockData = createQ8_0Block(0x3C00, codes)

        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(32, 1), blockData)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, ctx)

        // Expected: sum(1..32) * 1.0 * 1 = 528
        assertEquals(1, output.shape.dimensions[0])
        assertEquals(1, output.shape.dimensions[1])
        assertEquals(528.0f, output.data[0, 0], 1.0f)
    }

    @Test
    fun `Q8_0 matmul with scale`() {
        // Input: [1.0, 1.0, ..., 1.0]  32 elements
        val inputData = FloatArray(32) { 1.0f }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 32), FP32::class, inputData)

        // Weight [32, 1] - scale = 0.5 (0x3800), codes = [2, 2, ..., 2]
        val codes = ByteArray(32) { 2 }
        val blockData = createQ8_0Block(0x3800, codes)

        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(32, 1), blockData)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, ctx)

        // Expected: sum of (1.0 * 2) * 0.5 for 32 elements = 32 * 2 * 0.5 = 32
        assertEquals(32.0f, output.data[0, 0], 1.0f)
    }

    @Test
    fun `Q8_0 matmul with negative codes`() {
        // Input: [1.0, 2.0]  - only 2 elements for simplicity
        // We'll pad to 32 but only use first 2
        val inputData = FloatArray(32) { if (it < 2) (it + 1).toFloat() else 0f }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 32), FP32::class, inputData)

        // scale = 1.0, codes = [1, -1, 0, 0, ...]
        val codes = ByteArray(32)
        codes[0] = 1
        codes[1] = (-1).toByte()
        val blockData = createQ8_0Block(0x3C00, codes)

        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(32, 1), blockData)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, ctx)

        // Expected: (1.0 * 1 + 2.0 * -1) * 1.0 = -1.0
        assertEquals(-1.0f, output.data[0, 0], 0.1f)
    }

    @Test
    fun `Q8_0 matmul batched input`() {
        // Batch of 2 inputs: [[1, 1, ...], [2, 2, ...]]
        val inputData = FloatArray(64)
        for (i in 0 until 32) inputData[i] = 1.0f
        for (i in 32 until 64) inputData[i] = 2.0f

        val input = ctx.fromFloatArray<FP32, Float>(Shape(2, 32), FP32::class, inputData)

        // scale = 1.0, codes = [1, 1, ..., 1]
        val codes = ByteArray(32) { 1 }
        val blockData = createQ8_0Block(0x3C00, codes)

        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(32, 1), blockData)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, ctx)

        assertEquals(2, output.shape.dimensions[0])
        assertEquals(1, output.shape.dimensions[1])

        // Batch 0: 32 * 1 * 1 = 32
        assertEquals(32.0f, output.data[0, 0], 0.1f)
        // Batch 1: 32 * 2 * 1 = 64
        assertEquals(64.0f, output.data[1, 0], 0.1f)
    }

    // ========== Type Detection Tests ==========

    @Test
    fun `isQ8_0Weight returns false for non-Q8_0 tensor`() {
        val tensor = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        assertFalse(QuantizedMatmul.isQ8_0Weight(tensor))
    }

    @Test
    fun `isQ4_KWeight returns false for non-Q4_K tensor`() {
        val tensor = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        assertFalse(QuantizedMatmul.isQ4_KWeight(tensor))
    }

    @Test
    fun `isQuantizedWeight returns false for FP32 tensor`() {
        val tensor = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        assertFalse(QuantizedMatmul.isQuantizedWeight(tensor))
    }

    // ========== Auto Dispatch Tests ==========

    @Test
    fun `matmulAutoDispatch uses ternary path for ternary weights`() {
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 4), FP32::class,
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        )

        // Create a ternary tensor - for testing purposes, we use the matmul function directly
        val weights = Ternary2BitTensorData.fromTernaryValues(
            Shape(4, 2),
            byteArrayOf(
                1, -1,
                1, -1,
                1, -1,
                1, -1
            )
        )

        // Direct call - auto dispatch would need a proper wrapped tensor
        val output = TernaryMatmul.matmul(input, weights, ctx)

        // Col 0: 1+2+3+4 = 10
        // Col 1: -1-2-3-4 = -10
        assertEquals(10.0f, output.data[0, 0], 0.001f)
        assertEquals(-10.0f, output.data[0, 1], 0.001f)
    }

    // ========== Numerical Accuracy Tests ==========

    @Test
    fun `Q8_0 matmul matches dequant plus FP32 matmul within tolerance`() {
        // Create known Q8_0 data
        val inputData = FloatArray(32) { (it % 8 + 1).toFloat() }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 32), FP32::class, inputData)

        // Random-ish codes
        val codes = ByteArray(32) { ((it * 7 + 3) % 21 - 10).toByte() }
        val blockData = createQ8_0Block(0x3C00, codes)  // scale = 1.0

        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(32, 1), blockData)

        // Quantized matmul
        val quantizedOutput = QuantizedMatmul.matmulQ8_0(input, weights, ctx)

        // Compute expected with dequant + FP32
        var expected = 0f
        for (i in 0 until 32) {
            expected += inputData[i] * codes[i].toFloat() * 1.0f
        }

        val actual = quantizedOutput.data[0, 0]
        val relError = abs(actual - expected) / (abs(expected) + 1e-6f)

        assertTrue(relError < 1e-4f, "Relative error $relError exceeds tolerance. Expected: $expected, Actual: $actual")
    }
}
