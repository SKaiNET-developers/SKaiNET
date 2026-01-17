package sk.ainet.exec.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.avgPool2d
import sk.ainet.lang.types.FP32

class AvgPool2dOpsTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    @Test
    fun avgPool2d_basic_2x2_kernel() {
        // Input: 1 batch, 1 channel, 4x4 spatial
        // Shape: [1, 1, 4, 4]
        val input = tensor(
            Shape(1, 1, 4, 4),
            floatArrayOf(
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
            )
        )

        val output = input.avgPool2d(kernelSize = 2 to 2, stride = 2 to 2)

        // Output should be [1, 1, 2, 2]
        assertEquals(Shape(1, 1, 2, 2), output.shape)

        // Top-left: avg(1, 2, 5, 6) = 3.5
        assertEquals(3.5f, output.data[0, 0, 0, 0], 1e-6f)
        // Top-right: avg(3, 4, 7, 8) = 5.5
        assertEquals(5.5f, output.data[0, 0, 0, 1], 1e-6f)
        // Bottom-left: avg(9, 10, 13, 14) = 11.5
        assertEquals(11.5f, output.data[0, 0, 1, 0], 1e-6f)
        // Bottom-right: avg(11, 12, 15, 16) = 13.5
        assertEquals(13.5f, output.data[0, 0, 1, 1], 1e-6f)
    }

    @Test
    fun avgPool2d_3x3_kernel_stride_1() {
        // Input: 1 batch, 1 channel, 4x4 spatial
        val input = tensor(
            Shape(1, 1, 4, 4),
            floatArrayOf(
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
            )
        )

        val output = input.avgPool2d(kernelSize = 3 to 3, stride = 1 to 1)

        // Output should be [1, 1, 2, 2]
        assertEquals(Shape(1, 1, 2, 2), output.shape)

        // Top-left: avg(1,2,3,5,6,7,9,10,11) = 54/9 = 6.0
        assertEquals(6.0f, output.data[0, 0, 0, 0], 1e-6f)
        // Top-right: avg(2,3,4,6,7,8,10,11,12) = 63/9 = 7.0
        assertEquals(7.0f, output.data[0, 0, 0, 1], 1e-6f)
        // Bottom-left: avg(5,6,7,9,10,11,13,14,15) = 90/9 = 10.0
        assertEquals(10.0f, output.data[0, 0, 1, 0], 1e-6f)
        // Bottom-right: avg(6,7,8,10,11,12,14,15,16) = 99/9 = 11.0
        assertEquals(11.0f, output.data[0, 0, 1, 1], 1e-6f)
    }

    @Test
    fun avgPool2d_multiple_channels() {
        // Input: 1 batch, 2 channels, 2x2 spatial
        val input = tensor(
            Shape(1, 2, 2, 2),
            floatArrayOf(
                // Channel 0
                1f, 2f,
                3f, 4f,
                // Channel 1
                5f, 6f,
                7f, 8f
            )
        )

        val output = input.avgPool2d(kernelSize = 2 to 2, stride = 2 to 2)

        // Output should be [1, 2, 1, 1]
        assertEquals(Shape(1, 2, 1, 1), output.shape)

        // Channel 0: avg(1, 2, 3, 4) = 2.5
        assertEquals(2.5f, output.data[0, 0, 0, 0], 1e-6f)
        // Channel 1: avg(5, 6, 7, 8) = 6.5
        assertEquals(6.5f, output.data[0, 1, 0, 0], 1e-6f)
    }

    @Test
    fun avgPool2d_batch_processing() {
        // Input: 2 batches, 1 channel, 2x2 spatial
        val input = tensor(
            Shape(2, 1, 2, 2),
            floatArrayOf(
                // Batch 0
                1f, 2f,
                3f, 4f,
                // Batch 1
                10f, 20f,
                30f, 40f
            )
        )

        val output = input.avgPool2d(kernelSize = 2 to 2, stride = 2 to 2)

        // Output should be [2, 1, 1, 1]
        assertEquals(Shape(2, 1, 1, 1), output.shape)

        // Batch 0: avg(1, 2, 3, 4) = 2.5
        assertEquals(2.5f, output.data[0, 0, 0, 0], 1e-6f)
        // Batch 1: avg(10, 20, 30, 40) = 25.0
        assertEquals(25.0f, output.data[1, 0, 0, 0], 1e-6f)
    }

    @Test
    fun avgPool2d_with_padding() {
        // Input: 1 batch, 1 channel, 2x2 spatial
        val input = tensor(
            Shape(1, 1, 2, 2),
            floatArrayOf(
                1f, 2f,
                3f, 4f
            )
        )

        // With padding=1, the input becomes 4x4 (padded with zeros)
        // With 2x2 kernel and stride 2, output is 2x2
        val output = input.avgPool2d(kernelSize = 2 to 2, stride = 2 to 2, padding = 1 to 1, countIncludePad = true)

        // Output should be [1, 1, 2, 2]
        assertEquals(Shape(1, 1, 2, 2), output.shape)

        // Top-left window: [0,0,0,1] -> avg = 0.25 (with countIncludePad=true)
        assertEquals(0.25f, output.data[0, 0, 0, 0], 1e-6f)
    }

    @Test
    fun avgPool2d_count_exclude_pad() {
        // Input: 1 batch, 1 channel, 2x2 spatial
        val input = tensor(
            Shape(1, 1, 2, 2),
            floatArrayOf(
                4f, 4f,
                4f, 4f
            )
        )

        // With padding=1, the input becomes 4x4 (padded with zeros)
        // With countIncludePad=false, only non-padded values are counted
        val output = input.avgPool2d(kernelSize = 2 to 2, stride = 2 to 2, padding = 1 to 1, countIncludePad = false)

        // Output should be [1, 1, 2, 2]
        assertEquals(Shape(1, 1, 2, 2), output.shape)

        // Top-left window: only 1 actual value (4), so avg = 4
        assertEquals(4.0f, output.data[0, 0, 0, 0], 1e-6f)
    }

    @Test
    fun avgPool2d_global_average_pooling() {
        // Global average pooling: kernel size equals input spatial dimensions
        val input = tensor(
            Shape(1, 1, 3, 3),
            floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
            )
        )

        val output = input.avgPool2d(kernelSize = 3 to 3, stride = 1 to 1)

        // Output should be [1, 1, 1, 1]
        assertEquals(Shape(1, 1, 1, 1), output.shape)

        // avg(1, 2, 3, 4, 5, 6, 7, 8, 9) = 45/9 = 5.0
        assertEquals(5.0f, output.data[0, 0, 0, 0], 1e-6f)
    }

    @Test
    fun avgPool2d_non_square_kernel() {
        // Input: 1 batch, 1 channel, 4x4 spatial
        val input = tensor(
            Shape(1, 1, 4, 4),
            floatArrayOf(
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
            )
        )

        // 1x2 kernel with 1x2 stride
        val output = input.avgPool2d(kernelSize = 1 to 2, stride = 1 to 2)

        // Output should be [1, 1, 4, 2]
        assertEquals(Shape(1, 1, 4, 2), output.shape)

        // Row 0: avg(1,2)=1.5, avg(3,4)=3.5
        assertEquals(1.5f, output.data[0, 0, 0, 0], 1e-6f)
        assertEquals(3.5f, output.data[0, 0, 0, 1], 1e-6f)
    }

    @Test
    fun avgPool2d_preserves_dtype() {
        val input = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val output = input.avgPool2d(kernelSize = 2 to 2)

        assertEquals(FP32::class, output.dtype)
    }
}
