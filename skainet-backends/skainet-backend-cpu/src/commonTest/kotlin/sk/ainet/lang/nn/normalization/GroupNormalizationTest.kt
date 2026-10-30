package sk.ainet.lang.nn.normalization

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GroupNormalizationTest {

    private val ctx = DirectCpuExecutionContext()

    private fun assertAlmostEquals(expected: Float, actual: Float, eps: Float = 1e-5f, msg: String = "") {
        assertTrue(abs(expected - actual) <= eps, msg.ifEmpty { "Expected $expected, got $actual (diff=${abs(expected - actual)})" })
    }

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    /**
     * Reference GroupNorm with torch.nn.GroupNorm semantics, computed in Double:
     * per (sample, group) biased mean/variance over C/G * spatial elements,
     * then per-channel affine. Input is flat NCHW order.
     */
    private fun referenceGroupNorm(
        input: FloatArray,
        n: Int,
        c: Int,
        spatial: Int,
        groups: Int,
        eps: Double = 1e-5,
        gamma: FloatArray? = null,
        beta: FloatArray? = null,
    ): FloatArray {
        val channelsPerGroup = c / groups
        val groupSize = channelsPerGroup * spatial
        val out = FloatArray(input.size)
        for (b in 0 until n) {
            for (g in 0 until groups) {
                val indices = buildList {
                    for (ch in g * channelsPerGroup until (g + 1) * channelsPerGroup) {
                        for (s in 0 until spatial) add((b * c + ch) * spatial + s)
                    }
                }
                val mean = indices.sumOf { input[it].toDouble() } / groupSize
                val variance = indices.sumOf { (input[it] - mean) * (input[it] - mean) } / groupSize
                val denom = sqrt(variance + eps)
                for (i in indices) {
                    val ch = (i / spatial) % c
                    var v = (input[i] - mean) / denom
                    if (gamma != null) v = v * gamma[ch] + (beta?.get(ch)?.toDouble() ?: 0.0)
                    out[i] = v.toFloat()
                }
            }
        }
        return out
    }

    private fun assertFlatClose(expected: FloatArray, actual: Tensor<FP32, Float>, shape: Shape, eps: Float = 1e-5f) {
        assertEquals(shape, actual.shape)
        val dims = shape.dimensions
        var flat = 0
        val idx = IntArray(dims.size)
        while (flat < expected.size) {
            var rem = flat
            for (d in dims.indices.reversed()) {
                idx[d] = rem % dims[d]
                rem /= dims[d]
            }
            assertAlmostEquals(expected[flat], actual.data.get(*idx) as Float, eps, "Mismatch at flat index $flat")
            flat++
        }
    }

    @Test
    fun groupNorm_2d_noAffine_normalizesPerGroup() {
        // (N=1, C=4), G=2 -> groups [1,2] and [3,4], each mean +/- 0.5, var 0.25
        val data = floatArrayOf(1f, 2f, 3f, 4f)
        val input = tensor(Shape(1, 4), data)
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 4, affine = false, name = "gn")

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 1, c = 4, spatial = 1, groups = 2)
        assertFlatClose(expected, output, Shape(1, 4))
        // Sanity against hand math: (x - m) / sqrt(0.25 + 1e-5) ~= +/- 0.99998
        assertAlmostEquals(-0.99998f, output.data[0, 0] as Float, 1e-4f)
        assertAlmostEquals(0.99998f, output.data[0, 1] as Float, 1e-4f)
    }

    @Test
    fun groupNorm_4d_noAffine_matchesReference() {
        // (N=2, C=4, H=2, W=2), G=2 — varied values, groups span 2 channels * 4 spatial
        val data = FloatArray(32) { i -> (i * 7 % 13) - 6f + i * 0.25f }
        val input = tensor(Shape(2, 4, 2, 2), data)
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 4, affine = false, name = "gn4d")

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 2, c = 4, spatial = 4, groups = 2)
        assertFlatClose(expected, output, Shape(2, 4, 2, 2))
    }

    @Test
    fun groupNorm_4d_withAffine_appliesPerChannelGammaBeta() {
        // Distinct gamma/beta per channel: catches wrong broadcast axis.
        val data = FloatArray(16) { i -> i.toFloat() }
        val input = tensor(Shape(1, 4, 2, 2), data)
        val gammaArr = floatArrayOf(1f, 2f, 3f, 4f)
        val betaArr = floatArrayOf(0.5f, -0.5f, 1f, -1f)
        val gn = GroupNormalization<FP32, Float>(
            numGroups = 2,
            numChannels = 4,
            affine = true,
            name = "gn_affine",
            initGamma = tensor(Shape(4), gammaArr),
            initBeta = tensor(Shape(4), betaArr),
        )

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 1, c = 4, spatial = 4, groups = 2, gamma = gammaArr, beta = betaArr)
        assertFlatClose(expected, output, Shape(1, 4, 2, 2))
    }

    @Test
    fun groupNorm_defaultAffine_isIdentityScaleShift() {
        // Default placeholder params are gamma=1, beta=0 (the `groupNorm {}` DSL path):
        // output must equal the non-affine normalization.
        val data = floatArrayOf(2f, 4f, 6f, 8f, 1f, 3f, 5f, 7f)
        val withDefaultAffine = GroupNormalization<FP32, Float>(
            numGroups = 2, numChannels = 4, name = "gn_default", dtype = FP32::class,
        )
        val withoutAffine = GroupNormalization<FP32, Float>(
            numGroups = 2, numChannels = 4, affine = false, name = "gn_plain",
        )

        val a = withDefaultAffine.forward(tensor(Shape(2, 4), data), ctx)
        val b = withoutAffine.forward(tensor(Shape(2, 4), data), ctx)

        for (row in 0..1) {
            for (col in 0..3) {
                assertAlmostEquals(b.data[row, col] as Float, a.data[row, col] as Float, 1e-6f)
            }
        }
    }

    @Test
    fun groupNorm_singleGroup_normalizesWholeSample() {
        // G=1 normalizes over all C*spatial elements of each sample (LayerNorm-like).
        val data = floatArrayOf(
            1f, 2f, 3f, 4f, // sample 0
            10f, 20f, 30f, 40f, // sample 1
        )
        val input = tensor(Shape(2, 4), data)
        val gn = GroupNormalization<FP32, Float>(numGroups = 1, numChannels = 4, affine = false, name = "gn1")

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 2, c = 4, spatial = 1, groups = 1)
        assertFlatClose(expected, output, Shape(2, 4))
        // Each sample has ~zero mean after normalization
        for (row in 0..1) {
            var sum = 0f
            for (col in 0..3) sum += output.data[row, col] as Float
            assertAlmostEquals(0f, sum / 4f, 1e-5f, "Sample $row mean should be ~0")
        }
    }

    @Test
    fun groupNorm_groupsEqualChannels_isInstanceNorm() {
        // G=C: each channel normalized over its own spatial elements only.
        val data = floatArrayOf(
            1f, 3f, // channel 0: mean 2, var 1
            10f, 30f, // channel 1: mean 20, var 100
        )
        val input = tensor(Shape(1, 2, 2), data)
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 2, affine = false, name = "gnC")

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 1, c = 2, spatial = 2, groups = 2)
        assertFlatClose(expected, output, Shape(1, 2, 2))
        // Hand math: (1-2)/sqrt(1+1e-5) ~= -0.999995, (10-20)/sqrt(100+1e-5) ~= -1.0
        assertAlmostEquals(-0.999995f, output.data[0, 0, 0] as Float)
        assertAlmostEquals(-1f, output.data[0, 1, 0] as Float)
    }

    @Test
    fun groupNorm_constantInput_normalizesToZero() {
        // Zero variance: (x - mean) / sqrt(0 + eps) = 0 (eps prevents NaN/Inf).
        val input = tensor(Shape(1, 4), floatArrayOf(5f, 5f, 5f, 5f))
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 4, affine = false, name = "gn_const")

        val output = gn.forward(input, ctx)

        for (i in 0..3) {
            assertAlmostEquals(0f, output.data[0, i] as Float, 1e-2f, "Constant input should normalize to ~0")
        }
    }

    @Test
    fun groupNorm_customEps_entersDenominator() {
        // Large eps gives a visibly different result: (x - m) / sqrt(var + 1.0)
        val data = floatArrayOf(1f, 2f, 3f, 4f)
        val input = tensor(Shape(1, 4), data)
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 4, eps = 1.0, affine = false, name = "gn_eps")

        val output = gn.forward(input, ctx)

        val expected = referenceGroupNorm(data, n = 1, c = 4, spatial = 1, groups = 2, eps = 1.0)
        assertFlatClose(expected, output, Shape(1, 4))
        // var 0.25, denom sqrt(1.25): clearly distinct from the eps=1e-5 result
        assertAlmostEquals((-0.5 / sqrt(1.25)).toFloat(), output.data[0, 0] as Float)
    }

    @Test
    fun groupNorm_wrongChannelCount_throws() {
        val input = tensor(Shape(1, 6), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val gn = GroupNormalization<FP32, Float>(numGroups = 2, numChannels = 4, affine = false, name = "gn_bad")

        assertFailsWith<IllegalArgumentException> { gn.forward(input, ctx) }
    }

    @Test
    fun groupNorm_indivisibleGroups_throwsAtConstruction() {
        assertFailsWith<IllegalArgumentException> {
            GroupNormalization<FP32, Float>(numGroups = 3, numChannels = 4, name = "gn_indiv")
        }
    }
}
