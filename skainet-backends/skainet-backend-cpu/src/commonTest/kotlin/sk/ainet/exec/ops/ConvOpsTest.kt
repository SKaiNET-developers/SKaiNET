package sk.ainet.exec.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

class ConvOpsTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    // ========== Conv1d Tests ==========

    @Test
    fun conv1d_basic_no_padding_stride1() {
        // Input: (batch=1, channels=1, length=5) with values [1, 2, 3, 4, 5]
        val input = tensor(Shape(1, 1, 5), floatArrayOf(1f, 2f, 3f, 4f, 5f))
        // Weight: (out_channels=1, in_channels=1, kernel_size=3) with values [1, 1, 1]
        val weight = tensor(Shape(1, 1, 3), floatArrayOf(1f, 1f, 1f))

        val output = input.ops.conv1d(
            input = input,
            weight = weight,
            bias = null,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Output should be (1, 1, 3) with values [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        assertEquals(listOf(1, 1, 3), output.shape.dimensions.toList())
        assertEquals(6f, output.data[0, 0, 0], 1e-5f)
        assertEquals(9f, output.data[0, 0, 1], 1e-5f)
        assertEquals(12f, output.data[0, 0, 2], 1e-5f)
    }

    @Test
    fun conv1d_with_padding() {
        val input = tensor(Shape(1, 1, 5), floatArrayOf(1f, 2f, 3f, 4f, 5f))
        val weight = tensor(Shape(1, 1, 3), floatArrayOf(1f, 1f, 1f))

        val output = input.ops.conv1d(
            input = input,
            weight = weight,
            bias = null,
            stride = 1,
            padding = 1,  // Add padding to preserve length
            dilation = 1,
            groups = 1
        )

        // With padding=1, output length = 5
        assertEquals(listOf(1, 1, 5), output.shape.dimensions.toList())
    }

    @Test
    fun conv1d_with_stride2() {
        val input = tensor(Shape(1, 1, 7), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f))
        val weight = tensor(Shape(1, 1, 3), floatArrayOf(1f, 1f, 1f))

        val output = input.ops.conv1d(
            input = input,
            weight = weight,
            bias = null,
            stride = 2,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Output length = (7 - 3) / 2 + 1 = 3
        assertEquals(listOf(1, 1, 3), output.shape.dimensions.toList())
        assertEquals(6f, output.data[0, 0, 0], 1e-5f)  // 1+2+3
        assertEquals(12f, output.data[0, 0, 1], 1e-5f) // 3+4+5
        assertEquals(18f, output.data[0, 0, 2], 1e-5f) // 5+6+7
    }

    @Test
    fun conv1d_with_bias() {
        val input = tensor(Shape(1, 1, 5), floatArrayOf(1f, 2f, 3f, 4f, 5f))
        val weight = tensor(Shape(1, 1, 3), floatArrayOf(1f, 1f, 1f))
        val bias = tensor(Shape(1), floatArrayOf(10f))

        val output = input.ops.conv1d(
            input = input,
            weight = weight,
            bias = bias,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Output should have bias added: [6+10, 9+10, 12+10] = [16, 19, 22]
        assertEquals(16f, output.data[0, 0, 0], 1e-5f)
        assertEquals(19f, output.data[0, 0, 1], 1e-5f)
        assertEquals(22f, output.data[0, 0, 2], 1e-5f)
    }

    @Test
    fun conv1d_multiple_channels() {
        // Input: (batch=1, channels=2, length=4)
        val input = tensor(Shape(1, 2, 4), floatArrayOf(
            1f, 2f, 3f, 4f,   // Channel 0
            5f, 6f, 7f, 8f    // Channel 1
        ))
        // Weight: (out_channels=1, in_channels=2, kernel_size=2)
        val weight = tensor(Shape(1, 2, 2), floatArrayOf(
            1f, 1f,   // Filter for channel 0
            1f, 1f    // Filter for channel 1
        ))

        val output = input.ops.conv1d(
            input = input,
            weight = weight,
            bias = null,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Output: (1, 1, 3)
        // Position 0: (1+2) + (5+6) = 14
        // Position 1: (2+3) + (6+7) = 18
        // Position 2: (3+4) + (7+8) = 22
        assertEquals(listOf(1, 1, 3), output.shape.dimensions.toList())
        assertEquals(14f, output.data[0, 0, 0], 1e-5f)
        assertEquals(18f, output.data[0, 0, 1], 1e-5f)
        assertEquals(22f, output.data[0, 0, 2], 1e-5f)
    }

    // ========== Conv3d Tests ==========

    @Test
    fun conv3d_basic_shape() {
        // Input: (batch=1, channels=1, depth=4, height=4, width=4)
        val input = tensor(Shape(1, 1, 4, 4, 4), FloatArray(64) { 1f })
        // Weight: (out_channels=1, in_channels=1, kd=2, kh=2, kw=2)
        val weight = tensor(Shape(1, 1, 2, 2, 2), FloatArray(8) { 1f })

        val output = input.ops.conv3d(
            input = input,
            weight = weight,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Output shape: (1, 1, 3, 3, 3)
        assertEquals(listOf(1, 1, 3, 3, 3), output.shape.dimensions.toList())
        // Each output value should be 8 (sum of 2x2x2 block of 1s)
        assertEquals(8f, output.data[0, 0, 0, 0, 0], 1e-5f)
    }

    @Test
    fun conv3d_with_padding() {
        val input = tensor(Shape(1, 1, 4, 4, 4), FloatArray(64) { 1f })
        val weight = tensor(Shape(1, 1, 3, 3, 3), FloatArray(27) { 1f })

        val output = input.ops.conv3d(
            input = input,
            weight = weight,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(1, 1, 1),  // Same padding
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // With padding=1, output dimensions should be preserved
        assertEquals(listOf(1, 1, 4, 4, 4), output.shape.dimensions.toList())
    }

    @Test
    fun conv3d_with_stride() {
        val input = tensor(Shape(1, 1, 6, 6, 6), FloatArray(216) { 1f })
        val weight = tensor(Shape(1, 1, 2, 2, 2), FloatArray(8) { 1f })

        val output = input.ops.conv3d(
            input = input,
            weight = weight,
            bias = null,
            stride = Triple(2, 2, 2),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Output shape: (6-2)/2+1 = 3 for each dimension
        assertEquals(listOf(1, 1, 3, 3, 3), output.shape.dimensions.toList())
    }

    @Test
    fun conv3d_with_bias() {
        val input = tensor(Shape(1, 1, 3, 3, 3), FloatArray(27) { 1f })
        val weight = tensor(Shape(2, 1, 2, 2, 2), FloatArray(16) { 1f })
        val bias = tensor(Shape(2), floatArrayOf(5f, 10f))

        val output = input.ops.conv3d(
            input = input,
            weight = weight,
            bias = bias,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Output shape: (1, 2, 2, 2, 2)
        assertEquals(listOf(1, 2, 2, 2, 2), output.shape.dimensions.toList())
        // Each position = 8 (sum of 2x2x2 block) + bias
        assertEquals(13f, output.data[0, 0, 0, 0, 0], 1e-5f)  // 8 + 5
        assertEquals(18f, output.data[0, 1, 0, 0, 0], 1e-5f)  // 8 + 10
    }

    @Test
    fun conv3d_multiple_in_out_channels() {
        // Input: (batch=1, channels=2, depth=3, height=3, width=3)
        val input = tensor(Shape(1, 2, 3, 3, 3), FloatArray(54) { it.toFloat() })
        // Weight: (out_channels=3, in_channels=2, kd=2, kh=2, kw=2)
        val weight = tensor(Shape(3, 2, 2, 2, 2), FloatArray(48) { 1f })

        val output = input.ops.conv3d(
            input = input,
            weight = weight,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Output shape: (1, 3, 2, 2, 2)
        assertEquals(listOf(1, 3, 2, 2, 2), output.shape.dimensions.toList())
    }
}
