package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

class ConvVoidOpsShapeTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps()

    private fun tensor(shape: Shape): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(dataFactory.zeros(shape, FP32::class), FP32::class)

    // ========== Conv1d Shape Tests ==========

    @Test
    fun conv1d_no_padding_stride1_kernel3_returns_expected_shape() {
        // Input: NCL = (1, 3, 28)
        val x = tensor(Shape(1, 3, 28))
        // Weight: (out_channels=16, in_channels=3, kernel_length=3)
        val w = tensor(Shape(16, 3, 3))

        val y = ops.conv1d(
            input = x,
            weight = w,
            bias = null,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Expected: (1, 16, 28 - 3 + 1 = 26)
        assertEquals(listOf(1, 16, 26), y.shape.dimensions.toList())
    }

    @Test
    fun conv1d_same_padding_preserves_length() {
        val x = tensor(Shape(1, 3, 28))
        val w = tensor(Shape(16, 3, 5))

        val y = ops.conv1d(
            input = x,
            weight = w,
            bias = null,
            stride = 1,
            padding = 2,  // SAME-like padding for kernel=5
            dilation = 1,
            groups = 1
        )

        // Expected: (1, 16, 28)
        assertEquals(listOf(1, 16, 28), y.shape.dimensions.toList())
    }

    @Test
    fun conv1d_stride2_halves_length() {
        val x = tensor(Shape(1, 3, 28))
        val w = tensor(Shape(16, 3, 3))

        val y = ops.conv1d(
            input = x,
            weight = w,
            bias = null,
            stride = 2,
            padding = 0,
            dilation = 1,
            groups = 1
        )

        // Expected: (1, 16, (28 - 3) / 2 + 1 = 13)
        assertEquals(listOf(1, 16, 13), y.shape.dimensions.toList())
    }

    @Test
    fun conv1d_dilation2_effective_kernel() {
        val x = tensor(Shape(1, 3, 32))
        // 3-element kernel with dilation 2 → effective kernel = 1 + (3-1)*2 = 5
        val w = tensor(Shape(16, 3, 3))

        val y = ops.conv1d(
            input = x,
            weight = w,
            bias = null,
            stride = 1,
            padding = 0,
            dilation = 2,
            groups = 1
        )

        // Expected: (1, 16, 32 - 2*(3-1) - 1 + 1 = 28)
        assertEquals(listOf(1, 16, 28), y.shape.dimensions.toList())
    }

    // ========== Conv3d Shape Tests ==========

    @Test
    fun conv3d_no_padding_stride1_kernel3_returns_expected_shape() {
        // Input: NCDHW = (1, 3, 16, 16, 16)
        val x = tensor(Shape(1, 3, 16, 16, 16))
        // Weight: (out_channels=8, in_channels=3, kd=3, kh=3, kw=3)
        val w = tensor(Shape(8, 3, 3, 3, 3))

        val y = ops.conv3d(
            input = x,
            weight = w,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Expected: (1, 8, 14, 14, 14)
        assertEquals(listOf(1, 8, 14, 14, 14), y.shape.dimensions.toList())
    }

    @Test
    fun conv3d_same_padding_preserves_dims() {
        val x = tensor(Shape(1, 3, 16, 16, 16))
        val w = tensor(Shape(8, 3, 3, 3, 3))

        val y = ops.conv3d(
            input = x,
            weight = w,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(1, 1, 1),  // Same padding for 3x3x3
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Expected: (1, 8, 16, 16, 16)
        assertEquals(listOf(1, 8, 16, 16, 16), y.shape.dimensions.toList())
    }

    @Test
    fun conv3d_stride2_halves_dims() {
        val x = tensor(Shape(1, 3, 16, 16, 16))
        val w = tensor(Shape(8, 3, 2, 2, 2))

        val y = ops.conv3d(
            input = x,
            weight = w,
            bias = null,
            stride = Triple(2, 2, 2),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Expected: (1, 8, (16-2)/2+1=8, 8, 8)
        assertEquals(listOf(1, 8, 8, 8, 8), y.shape.dimensions.toList())
    }

    @Test
    fun conv3d_asymmetric_kernel() {
        val x = tensor(Shape(1, 3, 16, 24, 32))
        val w = tensor(Shape(8, 3, 3, 5, 7))

        val y = ops.conv3d(
            input = x,
            weight = w,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(1, 1, 1),
            groups = 1
        )

        // Expected: (1, 8, 16-3+1=14, 24-5+1=20, 32-7+1=26)
        assertEquals(listOf(1, 8, 14, 20, 26), y.shape.dimensions.toList())
    }

    @Test
    fun conv3d_dilation_affects_receptive_field() {
        val x = tensor(Shape(1, 3, 32, 32, 32))
        // 3x3x3 kernel with dilation 2 → effective kernel = 5x5x5
        val w = tensor(Shape(8, 3, 3, 3, 3))

        val y = ops.conv3d(
            input = x,
            weight = w,
            bias = null,
            stride = Triple(1, 1, 1),
            padding = Triple(0, 0, 0),
            dilation = Triple(2, 2, 2),
            groups = 1
        )

        // Effective kernel = 1 + (3-1)*2 = 5 for each dim
        // Expected: (1, 8, 32-4=28, 28, 28)
        assertEquals(listOf(1, 8, 28, 28, 28), y.shape.dimensions.toList())
    }
}
