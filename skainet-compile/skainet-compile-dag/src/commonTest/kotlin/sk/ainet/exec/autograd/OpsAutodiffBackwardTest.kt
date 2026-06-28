package sk.ainet.exec.autograd

import kotlin.math.abs
import kotlin.test.Test
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
 * Finite-difference backward parity for the autodiff-gap-closing work: the three previously
 * implemented-but-unwired activations (elu, leakyRelu, permute) and the newly-differentiable ops
 * (cos, sin, tril, gather, indexSelect, unfold, convTranspose1d). Each analytic gradient is compared
 * to central finite difference of a sum-reduced output. Tolerance is generous (FP32 noise);
 * correctness, not precision.
 */
class OpsAutodiffBackwardTest {

    private fun ctx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        val graph = DefaultComputeGraph()
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
            computeGraph = graph,
            baseSink = GraphSink(graph),
        )
    }

    private fun floatTensor(c: DefaultGraphExecutionContext, shape: Shape, values: FloatArray): Tensor<FP32, Float> =
        c.fromFloatArray(shape, FP32::class, values)

    private fun intTensor(c: DefaultGraphExecutionContext, shape: Shape, values: IntArray): Tensor<DType, Any> {
        @Suppress("UNCHECKED_CAST")
        return c.fromIntArray<Int32, Int>(shape, Int32::class, values) as Tensor<DType, Any>
    }

    private fun buf(t: Tensor<*, *>): FloatArray = (t.data as FloatArrayTensorData<*>).buffer

    private fun FloatArray.sumElems(): Float {
        var s = 0f
        for (v in this) s += v
        return s
    }

    private fun assertGradMatchesFiniteDiff(
        xShape: Shape,
        x0: FloatArray,
        eps: Float = 1e-3f,
        tol: Float = 3e-2f,
        f: (DefaultGraphExecutionContext, Tensor<FP32, Float>) -> Tensor<FP32, Float>,
    ) {
        val ctx = ctx()
        val x = floatTensor(ctx, xShape, x0.copyOf()).withRequiresGrad()
        val pair = ctx.record {
            val out = f(this, x)
            out.ops.sum(out)
        }
        val sumOutput = pair.second
        val tape = pair.first as DefaultGradientTape
        tape.computeGradients(targets = listOf(sumOutput), sources = listOf(x))
        val analyticGrad = x.grad
        assertNotNull(analyticGrad, "tape should populate x.grad")
        val analytic = buf(analyticGrad)

        for (i in x0.indices) {
            val xPlus = x0.copyOf().also { it[i] += eps }
            val xMinus = x0.copyOf().also { it[i] -= eps }
            val ctxPlus = ctx()
            val ctxMinus = ctx()
            val fPlus = buf(f(ctxPlus, floatTensor(ctxPlus, xShape, xPlus))).sumElems()
            val fMinus = buf(f(ctxMinus, floatTensor(ctxMinus, xShape, xMinus))).sumElems()
            val fdGrad = (fPlus - fMinus) / (2 * eps)
            val diff = abs(analytic[i] - fdGrad)
            assertTrue(diff <= tol, "[$i] analytic=${analytic[i]} fd=$fdGrad diff=$diff tol=$tol")
        }
    }

    // ── previously implemented-but-unwired (the silent-grad bug) ──────────────

    @Test
    fun elu_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(Shape(6), floatArrayOf(-1.5f, -0.4f, 0.3f, 0.9f, -0.7f, 1.2f), tol = 1e-2f) { _, x ->
            x.ops.elu(x, alpha = 1.0f)
        }
    }

    @Test
    fun leakyRelu_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(Shape(6), floatArrayOf(-1.5f, -0.4f, 0.3f, 0.9f, -0.7f, 1.2f), tol = 1e-2f) { _, x ->
            x.ops.leakyRelu(x, negativeSlope = 0.1f)
        }
    }

    @Test
    fun permute_backward_routes_axes_inverse() {
        // sum(w ⊙ permute(x)) — the constant weight makes the upstream non-uniform, so a wrong
        // inverse-axes would fail (a plain sum(permute(x)) has all-ones grad and can't detect it).
        assertGradMatchesFiniteDiff(Shape(2, 3), FloatArray(6) { (it - 2) * 0.3f }, tol = 1e-2f) { c, x ->
            val p = x.ops.permute(x, intArrayOf(1, 0)) // [2,3] -> [3,2]
            val w = floatTensor(c, Shape(3, 2), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
            x.ops.multiply(p, w)
        }
    }

    // ── trivial new diffs ─────────────────────────────────────────────────────

    @Test
    fun sin_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(Shape(5), floatArrayOf(-1f, -0.3f, 0.2f, 0.8f, 1.4f), tol = 1e-2f) { _, x -> x.ops.sin(x) }
    }

    @Test
    fun cos_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(Shape(5), floatArrayOf(-1f, -0.3f, 0.2f, 0.8f, 1.4f), tol = 1e-2f) { _, x -> x.ops.cos(x) }
    }

    @Test
    fun tril_backward_masks_upper_triangle() {
        // grad of sum(tril(x)) is the lower-triangular mask — position-dependent, so a no-op
        // backward (passing the full upstream) would fail.
        assertGradMatchesFiniteDiff(Shape(3, 3), FloatArray(9) { (it - 4) * 0.2f }, tol = 1e-2f) { _, x -> x.ops.tril(x, 0) }
    }

    // ── structural new diffs (scatter-add / fold / conv-transpose) ─────────────

    @Test
    fun gather_backward_scatter_adds_rows() {
        // table [vocab=4, emb=3], indices [0,2,2,1] -> row gradients = gather counts (1,1,2,0).
        assertGradMatchesFiniteDiff(Shape(4, 3), FloatArray(12) { (it - 6) * 0.1f }) { c, x ->
            val idx = intTensor(c, Shape(4), intArrayOf(0, 2, 2, 1))
            x.ops.gather(x, idx, dim = 0)
        }
    }

    @Test
    fun indexSelect_backward_scatter_adds_along_dim() {
        // x [3,4], dim=1, indices [0,2,2] -> col gradients = select counts (1,0,2,0).
        assertGradMatchesFiniteDiff(Shape(3, 4), FloatArray(12) { (it - 6) * 0.1f }) { c, x ->
            val idx = intTensor(c, Shape(3), intArrayOf(0, 2, 2))
            x.ops.indexSelect(x, idx, dim = 1)
        }
    }

    @Test
    fun unfold_backward_folds_overlapping_windows() {
        // x [6], size 3, step 1 -> 4 windows; each element's grad = number of windows covering it.
        assertGradMatchesFiniteDiff(Shape(6), FloatArray(6) { (it - 3) * 0.25f }) { _, x ->
            x.ops.unfold(x, dim = 0, size = 3, step = 1)
        }
    }

    @Test
    fun convTranspose1d_backward_matches_finite_diff() {
        // input [1,1,3], weight [in=1, outPerGroup=1, k=2], stride 1, padding 0.
        val w = floatArrayOf(0.5f, -1.2f)
        assertGradMatchesFiniteDiff(Shape(1, 1, 3), floatArrayOf(0.3f, -0.2f, 0.7f)) { c, x ->
            val wT = floatTensor(c, Shape(1, 1, 2), w)
            x.ops.convTranspose1d(x, wT, null, stride = 1, padding = 0, outputPadding = 0, dilation = 1, groups = 1)
        }
    }
}
