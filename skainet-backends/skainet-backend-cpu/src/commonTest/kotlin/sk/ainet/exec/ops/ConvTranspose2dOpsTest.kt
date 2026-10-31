package sk.ainet.exec.ops

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.TransposedConv2d
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

class ConvTranspose2dOpsTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    /**
     * Reference conv_transpose2d with torch.nn.functional.conv_transpose2d semantics,
     * accumulated in Double. Input is flat NCHW, weight is flat [C_in, C_out/groups, kH, kW].
     * out[b, oc, ih*sH - pH + kh*dH, iw*sW - pW + kw*dW] += in[b, ic, ih, iw] * w[ic, oc, kh, kw]
     */
    private fun referenceConvTranspose2d(
        input: FloatArray,
        n: Int,
        cIn: Int,
        inH: Int,
        inW: Int,
        weight: FloatArray,
        cOutPerGroup: Int,
        kH: Int,
        kW: Int,
        bias: FloatArray? = null,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        outputPadding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1,
    ): Pair<FloatArray, Shape> {
        val (sH, sW) = stride
        val (pH, pW) = padding
        val (opH, opW) = outputPadding
        val (dH, dW) = dilation
        val outH = (inH - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1
        val outW = (inW - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1
        val cOut = cOutPerGroup * groups
        val inChPerGroup = cIn / groups
        val out = DoubleArray(n * cOut * outH * outW)
        for (b in 0 until n) {
            for (g in 0 until groups) {
                for (ic in 0 until inChPerGroup) {
                    val inCh = g * inChPerGroup + ic
                    for (oc in 0 until cOutPerGroup) {
                        val outCh = g * cOutPerGroup + oc
                        for (ih in 0 until inH) {
                            for (iw in 0 until inW) {
                                val x = input[((b * cIn + inCh) * inH + ih) * inW + iw].toDouble()
                                for (kh in 0 until kH) {
                                    val oh = ih * sH - pH + kh * dH
                                    if (oh < 0 || oh >= outH) continue
                                    for (kw in 0 until kW) {
                                        val ow = iw * sW - pW + kw * dW
                                        if (ow < 0 || ow >= outW) continue
                                        val w = weight[((inCh * cOutPerGroup + oc) * kH + kh) * kW + kw].toDouble()
                                        out[((b * cOut + outCh) * outH + oh) * outW + ow] += x * w
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (bias != null) {
            for (b in 0 until n) {
                for (oc in 0 until cOut) {
                    val base = (b * cOut + oc) * outH * outW
                    for (i in 0 until outH * outW) out[base + i] += bias[oc].toDouble()
                }
            }
        }
        return FloatArray(out.size) { out[it].toFloat() } to Shape(n, cOut, outH, outW)
    }

    private fun assertFlatClose(expected: FloatArray, actual: Tensor<FP32, Float>, shape: Shape, eps: Float = 1e-4f) {
        assertEquals(shape, actual.shape, "shape mismatch")
        val dims = shape.dimensions
        val idx = IntArray(dims.size)
        for (flat in expected.indices) {
            var rem = flat
            for (d in dims.indices.reversed()) {
                idx[d] = rem % dims[d]
                rem /= dims[d]
            }
            val a = actual.data.get(*idx) as Float
            assertTrue(abs(expected[flat] - a) <= eps, "Mismatch at flat index $flat: expected ${expected[flat]}, got $a")
        }
    }

    // ========== Hand-verified basics ==========

    @Test
    fun convTranspose2d_basic_k2_s1_matchesHandComputation() {
        // input [[1,2],[3,4]], ones kernel 2x2, stride 1, no padding -> full scatter sum:
        // [[1,3,2],[4,10,6],[3,7,4]]
        val input = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val weight = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 1f, 1f, 1f))

        val output = input.ops.convTranspose2d(input, weight)

        assertFlatClose(
            floatArrayOf(1f, 3f, 2f, 4f, 10f, 6f, 3f, 7f, 4f),
            output,
            Shape(1, 1, 3, 3)
        )
    }

    @Test
    fun convTranspose2d_stride2_k2_stampsNonOverlappingKernels() {
        // k == stride: each input pixel stamps the kernel scaled by its value, no overlap.
        val input = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val weight = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f))

        val output = input.ops.convTranspose2d(input, weight, stride = 2 to 2)

        assertFlatClose(
            floatArrayOf(
                1f, 2f, 2f, 4f,
                3f, 4f, 6f, 8f,
                3f, 6f, 4f, 8f,
                9f, 12f, 12f, 16f,
            ),
            output,
            Shape(1, 1, 4, 4)
        )
    }

    // ========== Parameter grid vs double-precision reference ==========

    @Test
    fun convTranspose2d_multiChannel_stride2_padding1_outputPadding1_matchesReference() {
        val n = 1; val cIn = 2; val inH = 3; val inW = 3
        val cOut = 3; val kH = 3; val kW = 3
        val inputArr = FloatArray(n * cIn * inH * inW) { i -> ((i * 7) % 11) - 5f + i * 0.125f }
        val weightArr = FloatArray(cIn * cOut * kH * kW) { i -> ((i * 5) % 13) - 6f + i * 0.0625f }
        val input = tensor(Shape(n, cIn, inH, inW), inputArr)
        val weight = tensor(Shape(cIn, cOut, kH, kW), weightArr)

        val output = input.ops.convTranspose2d(
            input, weight,
            stride = 2 to 2, padding = 1 to 1, outputPadding = 1 to 1,
        )

        val (expected, shape) = referenceConvTranspose2d(
            inputArr, n, cIn, inH, inW, weightArr, cOut, kH, kW,
            stride = 2 to 2, padding = 1 to 1, outputPadding = 1 to 1,
        )
        assertFlatClose(expected, output, shape, eps = 1e-3f)
    }

    @Test
    fun convTranspose2d_asymmetricStrideAndPadding_matchesReference() {
        val n = 2; val cIn = 1; val inH = 2; val inW = 4
        val cOut = 2; val kH = 3; val kW = 2
        val inputArr = FloatArray(n * cIn * inH * inW) { i -> (i % 5) - 2f + i * 0.25f }
        val weightArr = FloatArray(cIn * cOut * kH * kW) { i -> ((i * 3) % 7) - 3f + i * 0.5f }
        val input = tensor(Shape(n, cIn, inH, inW), inputArr)
        val weight = tensor(Shape(cIn, cOut, kH, kW), weightArr)

        val output = input.ops.convTranspose2d(
            input, weight,
            stride = 2 to 3, padding = 1 to 0, outputPadding = 1 to 2,
        )

        val (expected, shape) = referenceConvTranspose2d(
            inputArr, n, cIn, inH, inW, weightArr, cOut, kH, kW,
            stride = 2 to 3, padding = 1 to 0, outputPadding = 1 to 2,
        )
        assertFlatClose(expected, output, shape, eps = 1e-3f)
    }

    @Test
    fun convTranspose2d_dilation2_matchesReference() {
        val n = 1; val cIn = 1; val inH = 3; val inW = 3
        val cOut = 1; val kH = 2; val kW = 2
        val inputArr = FloatArray(9) { i -> i + 1f }
        val weightArr = floatArrayOf(1f, -1f, 0.5f, 2f)
        val input = tensor(Shape(n, cIn, inH, inW), inputArr)
        val weight = tensor(Shape(cIn, cOut, kH, kW), weightArr)

        val output = input.ops.convTranspose2d(input, weight, dilation = 2 to 2)

        val (expected, shape) = referenceConvTranspose2d(
            inputArr, n, cIn, inH, inW, weightArr, cOut, kH, kW,
            dilation = 2 to 2,
        )
        assertFlatClose(expected, output, shape)
    }

    @Test
    fun convTranspose2d_groups2_matchesReference() {
        val n = 1; val cIn = 4; val inH = 2; val inW = 2
        val cOutPerGroup = 2; val kH = 2; val kW = 2; val groups = 2
        val inputArr = FloatArray(n * cIn * inH * inW) { i -> ((i * 3) % 5) - 2f + i * 0.1f }
        val weightArr = FloatArray(cIn * cOutPerGroup * kH * kW) { i -> ((i * 7) % 9) - 4f + i * 0.2f }
        val input = tensor(Shape(n, cIn, inH, inW), inputArr)
        val weight = tensor(Shape(cIn, cOutPerGroup, kH, kW), weightArr)

        val output = input.ops.convTranspose2d(input, weight, stride = 2 to 2, groups = groups)

        val (expected, shape) = referenceConvTranspose2d(
            inputArr, n, cIn, inH, inW, weightArr, cOutPerGroup, kH, kW,
            stride = 2 to 2, groups = groups,
        )
        assertFlatClose(expected, output, shape, eps = 1e-3f)
    }

    @Test
    fun convTranspose2d_bias_addsPerOutputChannel() {
        val input = tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val weight = tensor(Shape(1, 2, 2, 2), FloatArray(8) { 1f })
        val biasArr = floatArrayOf(10f, -10f)
        val bias = tensor(Shape(2), biasArr)

        val output = input.ops.convTranspose2d(input, weight, bias)

        val (expected, shape) = referenceConvTranspose2d(
            floatArrayOf(1f, 2f, 3f, 4f), 1, 1, 2, 2, FloatArray(8) { 1f }, 2, 2, 2,
            bias = biasArr,
        )
        assertFlatClose(expected, output, shape)
    }

    // ========== Layer integration ==========

    @Test
    fun transposedConv2dLayer_forward_matchesOutputSizeAndReference() {
        val inCh = 2; val outCh = 1; val k = 4 to 4
        val weightArr = FloatArray(inCh * outCh * 16) { i -> (i % 8) * 0.25f - 1f }
        val biasArr = floatArrayOf(0.5f)
        val layer = TransposedConv2d<FP32, Float>(
            inChannels = inCh,
            outChannels = outCh,
            kernelSize = k,
            stride = 2 to 2,
            padding = 1 to 1,
            name = "upconv",
            initWeights = tensor(Shape(inCh, outCh, 4, 4), weightArr),
            initBias = tensor(Shape(outCh), biasArr),
        )
        val inputArr = FloatArray(2 * 3 * 3) { i -> ((i * 5) % 7) - 3f }
        val input = tensor(Shape(1, inCh, 3, 3), inputArr)

        val output = layer.forward(input, ctx)

        val (expH, expW) = layer.outputSize(3 to 3)
        assertEquals(listOf(1, outCh, expH, expW), output.shape.dimensions.toList())
        val (expected, shape) = referenceConvTranspose2d(
            inputArr, 1, inCh, 3, 3, weightArr, outCh, 4, 4,
            bias = biasArr, stride = 2 to 2, padding = 1 to 1,
        )
        assertFlatClose(expected, output, shape, eps = 1e-3f)
    }

    // ========== Validation ==========

    @Test
    fun convTranspose2d_wrongWeightChannels_throws() {
        val input = tensor(Shape(1, 2, 2, 2), FloatArray(8) { 1f })
        val weight = tensor(Shape(3, 1, 2, 2), FloatArray(12) { 1f })
        assertFailsWith<IllegalArgumentException> { input.ops.convTranspose2d(input, weight) }
    }

    @Test
    fun convTranspose2d_outputPaddingNotLessThanStrideAndDilation_throws() {
        val input = tensor(Shape(1, 1, 2, 2), FloatArray(4) { 1f })
        val weight = tensor(Shape(1, 1, 2, 2), FloatArray(4) { 1f })
        assertFailsWith<IllegalArgumentException> {
            input.ops.convTranspose2d(input, weight, stride = 1 to 1, outputPadding = 1 to 1)
        }
    }

    @Test
    fun convTranspose2d_non4dInput_throws() {
        val input = tensor(Shape(1, 1, 4), FloatArray(4) { 1f })
        val weight = tensor(Shape(1, 1, 2, 2), FloatArray(4) { 1f })
        assertFailsWith<IllegalArgumentException> { input.ops.convTranspose2d(input, weight) }
    }
}
