package sk.ainet.exec.ops

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8

/**
 * `convTranspose1d` on the CPU backend per dtype (#1421): FP32, FP16, BF16 and Int32 compute,
 * everything else is rejected with a dtype error. Expected values are from
 * `torch.nn.functional.conv_transpose1d` (PyTorch 2.4.1). The float inputs are multiples of 1/4
 * with exactly representable results, so FP16 and BF16 must reproduce them without tolerance.
 */
class ConvTranspose1dOpsTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    // x [2,4,3], w [4,3,3], bias [6]; stride 2, padding 1, outputPadding 1, dilation 2, groups 2 -> [2,6,8]
    private val x = floatArrayOf(0.25f, -0.75f, 0.0f, 0.25f, 1.5f, 1.0f, -1.75f, 0.25f, -0.25f, -1.5f, -1.0f, -1.25f, 0.25f, 1.75f, -0.5f, 1.5f, 1.75f, 1.5f, -0.5f, 2.0f, 1.0f, -1.25f, 0.25f, 0.25f)
    private val w = floatArrayOf(-2.0f, 1.5f, 0.75f, 0.5f, -1.5f, 2.0f, 0.0f, -0.5f, 0.25f, -1.25f, 2.0f, -1.75f, 0.25f, -0.5f, 0.25f, 0.5f, 0.0f, 1.25f, -0.5f, -0.25f, -1.75f, 0.5f, -1.75f, 1.5f, 0.25f, 1.0f, -0.75f, -1.75f, 0.5f, -1.25f, -2.0f, -0.25f, -1.25f, 1.25f, -1.0f, 1.0f)
    private val bias = floatArrayOf(0.0f, -0.75f, 1.75f, -2.0f, -2.0f, -1.75f)
    private val expected = floatArrayOf(0.0f, 0.5f, 0.0f, 0.375f, 0.0f, -1.1875f, 0.0f, -1.75f, -0.75f, -1.25f, -0.75f, 0.4375f, -0.75f, -2.375f, -0.75f, -0.5f, 1.75f, 2.375f, 1.75f, 3.0f, 1.75f, 3.4375f, 1.75f, 3.0f, -2.0f, -0.6875f, -2.0f, 4.6875f, -2.0f, -1.75f, -2.0f, 0.0f, -2.0f, 3.5625f, -2.0f, -0.5625f, -2.0f, 0.375f, -2.0f, -0.8125f, -1.75f, -3.1875f, -1.75f, -2.3125f, -1.75f, -1.9375f, -1.75f, -2.8125f, 0.0f, -2.3125f, 0.0f, 2.8125f, 0.0f, 0.5f, 0.0f, -3.0f, -0.75f, -0.5625f, -0.75f, -3.25f, -0.75f, 3.1875f, -0.75f, -1.375f, 1.75f, 2.5f, 1.75f, 3.5625f, 1.75f, 4.625f, 1.75f, 3.5f, -2.0f, -3.9375f, -2.0f, -0.875f, -2.0f, -5.9375f, -2.0f, -4.0625f, -2.0f, -0.3125f, -2.0f, -4.75f, -2.0f, -1.125f, -2.0f, -0.8125f, -1.75f, -0.1875f, -1.75f, -0.3125f, -1.75f, -2.25f, -1.75f, -2.25f)

    @Test
    fun fp32_matches_pytorch() {
        val out = ctx.ops.convTranspose1d(
            ctx.fromFloatArray<FP32, Float>(Shape(2, 4, 3), FP32::class, x),
            ctx.fromFloatArray<FP32, Float>(Shape(4, 3, 3), FP32::class, w),
            ctx.fromFloatArray<FP32, Float>(Shape(6), FP32::class, bias),
            stride = 2, padding = 1, outputPadding = 1, dilation = 2, groups = 2,
        )
        assertEquals(Shape(2, 6, 8), out.shape)
        assertContentEquals(expected, out.data.copyToFloatArray())
    }

    @Test
    fun fp16_matches_pytorch() {
        val out = ctx.ops.convTranspose1d(
            ctx.fromFloatArray<FP16, Float>(Shape(2, 4, 3), FP16::class, x),
            ctx.fromFloatArray<FP16, Float>(Shape(4, 3, 3), FP16::class, w),
            ctx.fromFloatArray<FP16, Float>(Shape(6), FP16::class, bias),
            stride = 2, padding = 1, outputPadding = 1, dilation = 2, groups = 2,
        )
        assertEquals(FP16::class, out.dtype)
        assertEquals(Shape(2, 6, 8), out.shape)
        assertContentEquals(expected, out.data.copyToFloatArray())
    }

    @Test
    fun bf16_matches_pytorch() {
        val out = ctx.ops.convTranspose1d(
            ctx.fromFloatArray<BF16, Float>(Shape(2, 4, 3), BF16::class, x),
            ctx.fromFloatArray<BF16, Float>(Shape(4, 3, 3), BF16::class, w),
            ctx.fromFloatArray<BF16, Float>(Shape(6), BF16::class, bias),
            stride = 2, padding = 1, outputPadding = 1, dilation = 2, groups = 2,
        )
        assertEquals(BF16::class, out.dtype)
        assertContentEquals(expected, out.data.copyToFloatArray())
    }

    @Test
    fun bias_of_shape_1_c_1_equals_bias_of_shape_c() {
        val out = ctx.ops.convTranspose1d(
            ctx.fromFloatArray<FP32, Float>(Shape(2, 4, 3), FP32::class, x),
            ctx.fromFloatArray<FP32, Float>(Shape(4, 3, 3), FP32::class, w),
            ctx.fromFloatArray<FP32, Float>(Shape(1, 6, 1), FP32::class, bias),
            stride = 2, padding = 1, outputPadding = 1, dilation = 2, groups = 2,
        )
        assertContentEquals(expected, out.data.copyToFloatArray())
    }

    @Test
    fun int32_matches_pytorch() {
        // x [1,2,4], w [2,2,3], bias [2]; stride 2, padding 1 -> [1,2,7]
        val out = ctx.ops.convTranspose1d(
            ctx.fromIntArray<Int32, Int>(Shape(1, 2, 4), Int32::class, intArrayOf(-3, -2, 1, 4, -1, 1, -1, -1)),
            ctx.fromIntArray<Int32, Int>(Shape(2, 2, 3), Int32::class, intArrayOf(-2, -2, 2, 1, 3, 1, -1, -2, -2, -1, 2, -2)),
            ctx.fromIntArray<Int32, Int>(Shape(2), Int32::class, intArrayOf(1, -2)),
            stride = 2, padding = 1,
        )
        assertEquals(Int32::class, out.dtype)
        assertEquals(Shape(1, 2, 7), out.shape)
        val got = IntArray(14) { i -> out.data.get(0, i / 7, i % 7) as Int }
        assertContentEquals(intArrayOf(9, 0, 3, -6, 1, -2, -5, -13, -6, -6, -4, -1, 6, 8), got)
    }

    @Test
    fun int8_is_rejected_with_a_dtype_error() {
        val e = assertFailsWith<IllegalArgumentException> {
            ctx.ops.convTranspose1d(
                ctx.fromByteArray<Int8, Byte>(Shape(1, 1, 2), Int8::class, byteArrayOf(1, 2)),
                ctx.fromByteArray<Int8, Byte>(Shape(1, 1, 2), Int8::class, byteArrayOf(1, 1)),
            )
        }
        assertTrue(e.message!!.contains("Unsupported dtype for convTranspose1d"), e.message)
    }

    @Test
    fun invalid_arguments_are_rejected() {
        val x3 = ctx.fromFloatArray<FP32, Float>(Shape(1, 2, 3), FP32::class, FloatArray(6))
        val w3 = ctx.fromFloatArray<FP32, Float>(Shape(2, 1, 2), FP32::class, FloatArray(4))
        // weight C_in differs from the input channels
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.convTranspose1d(x3, ctx.fromFloatArray<FP32, Float>(Shape(3, 1, 2), FP32::class, FloatArray(6)))
        }
        assertFailsWith<IllegalArgumentException> { ctx.ops.convTranspose1d(x3, w3, stride = 0) }
        assertFailsWith<IllegalArgumentException> { ctx.ops.convTranspose1d(x3, w3, outputPadding = 1) }
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.convTranspose1d(x3, w3, bias = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, FloatArray(3)))
        }
        // input and weight of different dtypes
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.convTranspose1d(
                ctx.fromFloatArray<FP16, Float>(Shape(1, 2, 3), FP16::class, FloatArray(6)) as sk.ainet.lang.tensor.Tensor<FP32, Float>,
                w3,
            )
        }
    }
}
