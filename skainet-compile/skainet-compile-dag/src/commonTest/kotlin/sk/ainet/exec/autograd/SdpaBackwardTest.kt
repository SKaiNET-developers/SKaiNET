package sk.ainet.exec.autograd

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * `scaledDotProductAttentionBackward` (#1384): the analytic dQ, dK, dV (and dMask) against central
 * finite differences for every forward option — plain, causal, grouped-query, key padding, sliding
 * window with right context, dense mask, a KV-cache decode step, custom scale — and against PyTorch
 * autograd (`torch.nn.functional.scaled_dot_product_attention`, float64) on three fixed cases.
 */
class SdpaBackwardTest {

    private fun ctx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val graph = DefaultComputeGraph()
        return DefaultGraphExecutionContext(
            baseOps = DefaultCpuOps(dataFactory),
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
            computeGraph = graph,
            baseSink = GraphSink(graph),
        )
    }

    private fun DefaultGraphExecutionContext.floats(shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        fromFloatArray(shape, FP32::class, values)

    private fun buf(t: Tensor<*, *>): FloatArray = (t.data as FloatArrayTensorData<*>).buffer

    private fun randn(n: Int, seed: Int): FloatArray = Random(seed).let { r -> FloatArray(n) { (r.nextFloat() * 2f - 1f) * 1.5f } }

    /** `sum(sdpa(...) ⊙ w)`: the weights make every output element matter, so no gradient hides in a symmetric sum. */
    private class Sdpa(
        val shapes: List<Shape>,
        val weights: FloatArray,
        val scale: Float = 0f,
        val causal: Boolean = false,
        val keyLengths: IntArray? = null,
        val window: Int = 0,
        val rightContext: Int = 0,
    ) {
        /** inputs: q, k, v and, when [shapes] has four entries, the dense mask. */
        fun loss(c: DefaultGraphExecutionContext, inputs: List<Tensor<FP32, Float>>, w: Tensor<FP32, Float>): Tensor<FP32, Float> {
            val mask = inputs.getOrNull(3)
            @Suppress("UNCHECKED_CAST")
            val lengths = keyLengths?.let { c.fromIntArray<Int32, Int>(Shape(it.size), Int32::class, it) as Tensor<DType, Any> }
            val out = c.ops.scaledDotProductAttention(inputs[0], inputs[1], inputs[2], mask, scale, causal, lengths, window, rightContext)
            val weighted = out.ops.multiply(out, w)
            return weighted.ops.sum(weighted)
        }
    }

    /** Analytic gradients of every input of [case] at [values]. */
    private fun analytic(case: Sdpa, values: List<FloatArray>): List<FloatArray> {
        val c = ctx()
        val xs = case.shapes.indices.map { c.floats(case.shapes[it], values[it].copyOf()).withRequiresGrad() }
        val w = c.floats(case.shapes[0], case.weights)
        val (tape, loss) = c.record { case.loss(this as DefaultGraphExecutionContext, xs, w) }
        (tape as DefaultGradientTape).computeGradients(targets = listOf(loss), sources = xs)
        return xs.mapIndexed { i, x -> buf(assertNotNull(x.grad, "input $i has no gradient")) }
    }

    private fun lossAt(case: Sdpa, values: List<FloatArray>): Float {
        val c = ctx()
        val xs = case.shapes.indices.map { c.floats(case.shapes[it], values[it]) }
        return buf(case.loss(c, xs, c.floats(case.shapes[0], case.weights)))[0]
    }

    /** Every analytic gradient entry against central finite differences; returns the largest difference seen. */
    private fun assertFiniteDiff(case: Sdpa, values: List<FloatArray>, eps: Float = 1e-2f, tol: Float = 1e-2f): Float {
        val grads = analytic(case, values)
        var worst = 0f
        for (i in values.indices) for (j in values[i].indices) {
            val plus = values.mapIndexed { n, a -> if (n == i) a.copyOf().also { it[j] += eps } else a }
            val minus = values.mapIndexed { n, a -> if (n == i) a.copyOf().also { it[j] -= eps } else a }
            val fd = (lossAt(case, plus) - lossAt(case, minus)) / (2 * eps)
            val diff = abs(grads[i][j] - fd)
            worst = maxOf(worst, diff)
            assertTrue(diff <= tol, "input $i [$j]: analytic=${grads[i][j]} finite-diff=$fd diff=$diff tol=$tol")
        }
        return worst
    }

    private fun qkv(b: Int, h: Int, kvHeads: Int, seqQ: Int, seqKV: Int, headDim: Int): List<Shape> =
        listOf(Shape(b, h, seqQ, headDim), Shape(b, kvHeads, seqKV, headDim), Shape(b, kvHeads, seqKV, headDim))

    private fun values(shapes: List<Shape>, seed: Int): List<FloatArray> = shapes.mapIndexed { i, s -> randn(s.volume, seed + i) }

    // ── finite differences, one case per forward option ───────────────────────────────────────

    @Test
    fun plain_attention_matches_finite_diff() {
        val shapes = qkv(b = 2, h = 2, kvHeads = 2, seqQ = 3, seqKV = 3, headDim = 4)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 90)), values(shapes, 10))
    }

    @Test
    fun causal_attention_matches_finite_diff() {
        val shapes = qkv(b = 1, h = 2, kvHeads = 2, seqQ = 4, seqKV = 4, headDim = 3)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 91), causal = true), values(shapes, 20))
    }

    @Test
    fun grouped_query_attention_sums_head_groups_into_k_and_v() {
        val shapes = qkv(b = 1, h = 4, kvHeads = 2, seqQ = 3, seqKV = 3, headDim = 4)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 92), causal = true), values(shapes, 30))
    }

    @Test
    fun key_padding_gives_padded_keys_exactly_zero_gradient() {
        val shapes = qkv(b = 2, h = 2, kvHeads = 2, seqQ = 3, seqKV = 4, headDim = 3)
        val lengths = intArrayOf(3, 2)
        val case = Sdpa(shapes, randn(shapes[0].volume, 93), keyLengths = lengths)
        val vals = values(shapes, 40)
        assertFiniteDiff(case, vals)
        val (_, dK, dV) = analytic(case, vals)
        for (b in 0 until 2) for (h in 0 until 2) for (k in lengths[b] until 4) for (d in 0 until 3) {
            val i = ((b * 2 + h) * 4 + k) * 3 + d
            assertEquals(0f, dK[i], "dK[b=$b,h=$h,k=$k,d=$d] of a padded key")
            assertEquals(0f, dV[i], "dV[b=$b,h=$h,k=$k,d=$d] of a padded key")
        }
    }

    @Test
    fun sliding_window_with_right_context_matches_finite_diff() {
        val shapes = qkv(b = 1, h = 2, kvHeads = 2, seqQ = 5, seqKV = 5, headDim = 3)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 94), window = 2, rightContext = 1), values(shapes, 50))
    }

    @Test
    fun dense_mask_gets_a_gradient_and_broadcasts_over_heads() {
        val shapes = qkv(b = 2, h = 2, kvHeads = 2, seqQ = 3, seqKV = 3, headDim = 3) + Shape(2, 1, 3, 3)
        val vals = values(shapes, 60)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 95)), vals)
    }

    @Test
    fun dense_mask_per_kv_head_in_grouped_query_attention() {
        val shapes = qkv(b = 1, h = 4, kvHeads = 2, seqQ = 2, seqKV = 3, headDim = 3) + Shape(1, 2, 1, 3)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 96)), values(shapes, 70))
    }

    @Test
    fun decode_step_with_one_query_over_a_cache_matches_finite_diff() {
        val shapes = qkv(b = 1, h = 2, kvHeads = 1, seqQ = 1, seqKV = 5, headDim = 4)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 97), causal = true), values(shapes, 80))
    }

    @Test
    fun custom_scale_is_applied_to_dq_and_dk() {
        val shapes = qkv(b = 1, h = 2, kvHeads = 2, seqQ = 3, seqKV = 3, headDim = 4)
        assertFiniteDiff(Sdpa(shapes, randn(shapes[0].volume, 98), scale = 0.5f), values(shapes, 85))
    }

    // ── PyTorch autograd parity (float64 reference, generated by torch 2.5.1) ─────────────────

    private fun assertTorch(label: String, shapes: List<Shape>, q: FloatArray, k: FloatArray, v: FloatArray, w: FloatArray,
                            dQ: FloatArray, dK: FloatArray, dV: FloatArray, causal: Boolean, scale: Float = 0f, tol: Float = 2e-4f) {
        val got = analytic(Sdpa(shapes, w, scale = scale, causal = causal), listOf(q, k, v))
        for ((name, pair) in listOf("dQ" to (dQ to got[0]), "dK" to (dK to got[1]), "dV" to (dV to got[2]))) {
            val (want, have) = pair
            assertEquals(want.size, have.size, "$label $name size")
            for (i in want.indices) assertTrue(abs(want[i] - have[i]) <= tol, "$label $name[$i]: torch=${want[i]} skainet=${have[i]}")
        }
    }

    @Test
    fun matches_pytorch_plain() = Torch.Plain.run {
        assertTorch("plain", qkv(1, 2, 2, 3, 3, 4), q, k, v, w, dQ, dK, dV, causal = false)
    }

    @Test
    fun matches_pytorch_causal() = Torch.Causal.run {
        assertTorch("causal", qkv(1, 2, 2, 4, 4, 4), q, k, v, w, dQ, dK, dV, causal = true)
    }

    @Test
    fun matches_pytorch_grouped_query_causal_scaled() = Torch.GqaCausalScaled.run {
        assertTorch("gqa", qkv(1, 4, 2, 3, 3, 4), q, k, v, w, dQ, dK, dV, causal = true, scale = 0.5f)
    }

    /**
     * Reference gradients of `(sdpa(q, k, v) * w).sum()` from PyTorch 2.5.1, float64, `torch.manual_seed(1384)`;
     * `enable_gqa=True` for the grouped case. Row-major, shapes as in the test above.
     */
    private object Torch {
        object Plain {
            val q = floatArrayOf(0.2675929f, 1.411567f, -0.1011506f, 0.3230369f, -1.675892f, 0.4868996f, 2.726261f, -0.5353137f, 1.039413f, 0.0001392366f, 0.1926292f, -1.754218f, -0.7797833f, 1.243931f, -0.07230858f, -0.9458256f, -0.6858122f, 0.069565f, -1.728357f, 1.826367f, 0.2220952f, 0.4242508f, 0.7450087f, 0.5697452f)
            val k = floatArrayOf(0.5013323f, -0.2866618f, 1.103905f, 0.3208946f, 0.09062664f, -1.88496f, -0.1421603f, 0.8531001f, -0.2340906f, -0.4839675f, -0.9764906f, -0.728122f, -0.6217012f, 2.108704f, 0.7088581f, -0.7448576f, -1.748276f, 0.905459f, -1.07981f, -0.0810712f, -0.2192353f, -0.2226977f, 1.505503f, -0.6060776f)
            val v = floatArrayOf(0.9708158f, 1.556199f, -0.1814299f, -0.4377889f, 0.7130228f, 0.4296545f, 1.847022f, 0.26722f, -0.6324137f, 0.08507826f, 0.3509339f, 0.1849344f, -1.571065f, -0.2592146f, -0.960515f, -0.188378f, -2.57617f, -0.9815503f, -0.1900549f, 0.8855873f, -0.06818297f, 0.8406212f, -1.214263f, -0.5565868f)
            val w = floatArrayOf(-0.2467783f, -0.8568549f, 0.6836001f, 0.7255383f, 0.4093301f, -0.1812685f, -0.9924607f, 1.434568f, 0.1389276f, -0.4726173f, -1.097207f, -0.6673204f, -1.003371f, -0.8862868f, -1.265532f, 0.06454479f, 0.6696557f, 0.0715685f, -0.4430402f, 1.357646f, 0.04990109f, -1.543128f, 0.6105736f, 0.4715655f)
            val dQ = floatArrayOf(-0.1974019f, -0.2496873f, -0.5695557f, -0.1337727f, 0.01589591f, 0.07381187f, 0.04890416f, -0.02957778f, 0.03736139f, 0.1968742f, 0.1162673f, -0.08709103f, -0.1797418f, 0.123891f, -0.3034105f, 0.06290352f, -0.007120084f, -0.02268509f, -0.01044361f, 0.006248249f, -0.3714578f, 0.4399563f, -0.6375344f, 0.1049587f)
            val dK = floatArrayOf(-0.04782278f, -0.4407001f, 0.1687052f, -0.3145559f, -0.01681239f, 0.1638348f, -0.1649714f, 0.2879867f, 0.06463517f, 0.2768653f, -0.003733804f, 0.02656923f, 0.03087205f, 0.02830226f, 0.08479232f, 0.02543242f, -0.04902986f, 0.2421664f, 0.142122f, 0.02903788f, 0.01815781f, -0.2704686f, -0.2269143f, -0.0544703f)
            val dV = floatArrayOf(0.2485458f, -0.7154413f, -0.8286623f, 1.225317f, 0.02818142f, -0.2412318f, -0.1736651f, 0.1868334f, 0.02475213f, -0.5540675f, -0.40374f, 0.08063564f, -0.5010719f, -1.165915f, -0.4903299f, 0.3401799f, 0.2797954f, -0.5076518f, -0.696405f, 1.319442f, -0.06253787f, -0.6842803f, 0.08873683f, 0.2341345f)
        }

        object Causal {
            val q = floatArrayOf(-0.4916503f, -0.5389487f, -0.5767441f, 1.527768f, -0.9761692f, -0.8804677f, -2.38216f, 0.6664432f, -1.842302f, -0.6081567f, -0.1325497f, -1.158639f, 0.000725375f, 1.002904f, 0.5227017f, -2.444986f, -0.1151447f, 0.04876244f, 0.9931078f, 1.299705f, -0.1812805f, -0.2819617f, 0.1380914f, 0.8126997f, 0.6654037f, -0.1740248f, 0.7350452f, 0.8076161f, 0.5992458f, 1.126201f, 0.89685f, -1.890724f)
            val k = floatArrayOf(-0.2965695f, 0.1658968f, 0.001274851f, 0.1024414f, -0.9987526f, 1.280174f, 0.05974628f, -0.5729544f, 0.4300002f, 0.9034657f, 0.1883813f, 1.428727f, 1.104256f, 2.563296f, -0.05679508f, -0.07394058f, 0.2674591f, 0.7290081f, -0.7555979f, 0.3068895f, 0.5430296f, 0.2404021f, 1.225856f, -0.1206958f, -0.486159f, -2.397587f, -0.1283597f, -1.807623f, -0.8804631f, 0.004881665f, 1.871904f, -2.086527f)
            val v = floatArrayOf(1.635278f, -0.3255828f, -0.620495f, -0.2712025f, -0.3608688f, 0.7596877f, 0.1085253f, -0.2599764f, -1.352634f, -0.3399528f, 0.166733f, 0.7021374f, -1.118119f, 0.0542889f, -2.950635f, 0.2255886f, 0.2384639f, 0.6705991f, 0.78544f, -3.114786f, 0.4843569f, -0.546144f, 0.926424f, 0.5743018f, 2.801849f, 0.3078977f, -1.24524f, -0.6071633f, 0.3192707f, -0.137901f, 2.694265f, 2.236457f)
            val w = floatArrayOf(0.04640669f, -1.116565f, 1.09947f, -0.1892404f, -0.1091776f, -0.9821594f, -1.137273f, -2.112252f, 0.1419125f, 2.360541f, 0.6024354f, -0.7459311f, -0.2535677f, -0.5076545f, 1.231461f, 0.7545727f, -0.4003165f, 1.106979f, -1.327959f, -0.8141334f, 1.252254f, -0.6952007f, -0.32503f, -0.5761403f, -1.061178f, -0.7916225f, 0.5697017f, 0.1929754f, 1.278387f, -0.2564536f, -0.172122f, 0.8406162f)
            val dQ = floatArrayOf(0f, 0f, 0f, 0f, 0.1422063f, -0.2256638f, -0.01184165f, 0.1367814f, -0.2819649f, 0.3159145f, 0.006252877f, -0.3148601f, -0.6363526f, -0.4791033f, 0.04437827f, -0.06361784f, 5.938784e-17f, 1.618723e-16f, -1.677764e-16f, 6.814317e-17f, -0.03504716f, 0.06214109f, -0.2520021f, 0.05438047f, 0.2899925f, 0.6824186f, 0.4640558f, 0.4305233f, -0.1562029f, -0.2138765f, 0.1957545f, -0.31509f)
            val dK = floatArrayOf(0.2813104f, 0.001605902f, -0.4366112f, 0.3830595f, -0.4092687f, 0.2520733f, 0.5816558f, -1.185264f, 0.1282057f, 0.08834617f, 0.03321491f, -0.0316216f, -0.0002473782f, -0.3420254f, -0.1782595f, 0.8338258f, -0.120132f, -0.1457209f, -0.1144368f, 0.2615371f, 0.2116979f, -0.06415508f, 0.1815448f, 0.2290201f, -0.1464268f, 0.1067722f, -0.1492147f, -0.3174613f, 0.05486099f, 0.1031038f, 0.08210667f, -0.1730959f)
            val dV = floatArrayOf(-0.00252114f, -1.028908f, 0.7264782f, -1.628907f, -0.06266652f, 0.8798368f, 0.4632463f, -0.9700148f, 0.0002462512f, 0.1224269f, 0.07465291f, -0.01973699f, -0.1094847f, -0.2191936f, 0.5317167f, 0.3258072f, -0.01234861f, 0.5242475f, -1.337597f, -1.004809f, 0.2223879f, -0.8328272f, 0.1318366f, -0.05456945f, -0.07866927f, -0.1395932f, 0.07661259f, 0.08605212f, 0.9377764f, -0.1881247f, -0.1262622f, 0.6166444f)
        }

        object GqaCausalScaled {
            val q = floatArrayOf(-0.06110927f, -0.3452047f, -0.8787841f, -0.727814f, 0.1277452f, -0.4457449f, -0.9182072f, 0.4976269f, -0.3276936f, -0.184982f, 0.7957395f, -0.4093225f, -1.446154f, -0.9513799f, -0.1905794f, 0.1395682f, 0.4470229f, 0.1028677f, -0.5955917f, 1.142406f, -0.718825f, -2.255068f, 0.4683639f, -0.7382633f, -1.860136f, -1.002743f, -0.5541535f, 1.280307f, 0.6766555f, -1.185335f, 1.200517f, -0.03471418f, 0.2132596f, 0.04722201f, -0.8502766f, 0.3501975f, 0.7727007f, -2.039077f, -2.08312f, -1.072766f, -0.6677929f, 1.370682f, 0.5458515f, -2.224779f, 1.053937f, -0.5688137f, 1.548171f, -1.488023f)
            val k = floatArrayOf(2.018591f, -0.05519409f, -0.5033644f, 0.3380581f, 0.074922f, 1.910222f, 0.03126912f, -0.9849722f, -0.4488003f, 0.2485246f, -0.4897337f, -0.1820537f, 0.1074487f, 0.8746405f, 0.7097022f, -0.1927362f, -0.8289353f, -1.70231f, -1.002717f, 1.166588f, -0.2907353f, 1.135494f, -0.4035317f, -0.849929f)
            val v = floatArrayOf(-0.5467067f, -0.3381246f, -0.4230412f, 1.203924f, -1.465932f, 0.6048443f, -0.4126113f, 0.9579029f, -0.5995004f, 0.9532911f, 1.163246f, 0.2839137f, -1.491896f, 1.685945f, -0.5165589f, -1.346834f, 2.088984f, -0.1118536f, 0.6201755f, -0.9253777f, -0.05104136f, 0.683785f, 0.6090024f, -0.546514f)
            val w = floatArrayOf(0.7298373f, -0.3692693f, 0.1313081f, -1.430836f, -1.433704f, -1.145149f, -0.2478517f, 0.2417608f, 1.233094f, -0.003314593f, 0.6548823f, 1.060566f, 0.5140984f, -0.3680377f, 0.2167198f, 0.007885461f, 1.026577f, -0.380975f, -0.9757947f, -0.08412457f, -0.4994523f, -0.7525112f, 0.7750653f, -0.3387198f, -0.8815747f, 0.2042147f, -2.316481f, -0.2921494f, -1.543611f, 2.114498f, -1.616449f, 0.8422503f, 0.5974992f, -0.8138859f, 0.01642376f, -1.808418f, -0.1546724f, 0.3963216f, -0.000162008f, -0.004900547f, 2.316963f, -2.18078f, -1.052521f, -0.150756f, 1.025442f, 1.194332f, 0.09214855f, -1.802534f)
            val dQ = floatArrayOf(0f, 0f, 0f, 0f, -0.03145413f, 0.03180606f, 0.0086519f, -0.02141042f, 0.07085297f, -0.2991512f, -0.08853769f, 0.1686022f, 0f, 0f, 0f, 0f, 0.217625f, -0.2200599f, -0.05986081f, 0.1481345f, -0.134257f, -0.03312896f, -0.01432878f, -0.002028037f, 0f, 0f, 0f, 0f, 1.257494f, 3.460652f, 2.299651f, -1.825471f, -0.2454592f, -0.9211638f, -0.3791634f, 0.5799918f, 0f, 0f, 0f, 0f, -0.1554665f, -0.4278476f, -0.2843106f, 0.2256868f, 0.009732928f, -0.1422998f, 0.06578925f, 0.1398207f)
            val dK = floatArrayOf(0.06159543f, 0.115748f, -0.02318375f, 0.1291487f, 0.02782861f, 0.07726194f, -0.09553051f, -0.02975154f, -0.08942404f, -0.1930099f, 0.1187143f, -0.09939718f, 1.072315f, -1.873512f, 1.782968f, 0.1453452f, -0.9023448f, 1.81117f, -1.743977f, -0.2673907f, -0.1699705f, 0.06234227f, -0.03899151f, 0.1220455f)
            val dV = floatArrayOf(1.088163f, -2.116057f, -0.2304565f, -1.165766f, 0.3336714f, -0.4515641f, 0.08315176f, 0.4400944f, 0.1486161f, -0.4516351f, 0.7016339f, 0.182204f, 1.251841f, 0.01277365f, -4.02065f, -1.545289f, -0.3943006f, 0.7678278f, -0.8904103f, -0.5613586f, 0.5025052f, 0.1340986f, 0.03401926f, -1.109859f)
        }
    }
}
