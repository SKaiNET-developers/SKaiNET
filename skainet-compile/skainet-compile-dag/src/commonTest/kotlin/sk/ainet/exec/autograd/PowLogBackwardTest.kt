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
import sk.ainet.lang.tensor.pow
import sk.ainet.lang.tensor.log
import sk.ainet.lang.tensor.log2
import sk.ainet.lang.tensor.log10
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.FP32

/**
 * Tier C parity tests — every new backward formula compared to
 * finite-difference (`(f(x+ε) - f(x-ε)) / 2ε`). Tolerance is
 * deliberately generous (1e-2) to absorb FP32 noise; correctness,
 * not precision.
 */
class PowLogBackwardTest {

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

    private fun floatTensor(c: DefaultGraphExecutionContext, values: FloatArray): Tensor<FP32, Float> =
        c.fromFloatArray(Shape(values.size), FP32::class, values)

    private fun buf(t: Tensor<*, *>): FloatArray = (t.data as FloatArrayTensorData<*>).buffer

    /**
     * Verify analytic gradient (from the tape) against the central
     * finite-difference numerical gradient of [f] at each element of
     * [x0]. Each element-wise partial is checked separately by perturbing
     * that one element. Output is reduced to a scalar via sum-of-elements
     * inside [f] so the resulting Jacobian-vector product matches a
     * column of the Jacobian.
     */
    private fun assertGradMatchesFiniteDiff(
        x0: FloatArray,
        eps: Float = 1e-3f,
        tol: Float = 1e-2f,
        f: (DefaultGraphExecutionContext, Tensor<FP32, Float>) -> Tensor<FP32, Float>,
    ) {
        // 1. Compute analytic grad via the tape.
        val ctx = ctx()
        val x = floatTensor(ctx, x0.copyOf()).withRequiresGrad()
        val pair = ctx.record {
            val out = f(this, x)
            // Sum-reduce to a scalar so the gradient corresponds to df/dx
            // (kept inside the record block so the sum itself is taped).
            out.ops.sum(out)
        }
        val sumOutput = pair.second
        val tape = pair.first as DefaultGradientTape
        tape.computeGradients(targets = listOf(sumOutput), sources = listOf(x))
        val analyticGrad = x.grad
        assertNotNull(analyticGrad, "tape should populate x.grad")
        val analytic = buf(analyticGrad)

        // 2. Finite difference per-element.
        for (i in x0.indices) {
            val xPlus = x0.copyOf().also { it[i] += eps }
            val xMinus = x0.copyOf().also { it[i] -= eps }
            val ctxPlus = ctx()
            val ctxMinus = ctx()
            val fPlusOut = buf(f(ctxPlus, floatTensor(ctxPlus, xPlus)))
            val fMinusOut = buf(f(ctxMinus, floatTensor(ctxMinus, xMinus)))
            val fdGrad = (fPlusOut.sum() - fMinusOut.sum()) / (2 * eps)
            val diff = abs(analytic[i] - fdGrad)
            assertTrue(
                diff <= tol,
                "[$i] analytic=${analytic[i]} fd=$fdGrad diff=$diff tol=$tol  (x0=${x0.toList()})",
            )
        }
    }

    private fun FloatArray.sum(): Float {
        var s = 0f
        for (v in this) s += v
        return s
    }

    @Test
    fun powScalar_squared_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(0.5f, 1f, 1.5f, 2f, 3f)) { _, x -> x.pow(2) }
    }

    @Test
    fun powScalar_cubed_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(0.5f, 1f, 1.5f, 2f, 3f)) { _, x -> x.pow(3) }
    }

    @Test
    fun powScalar_real_exponent_1p5_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(0.5f, 1f, 2f, 4f)) { _, x -> x.pow(1.5f) }
    }

    @Test
    fun log_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(0.5f, 1f, 2f, 3f, 10f)) { _, x -> x.log() }
    }

    @Test
    fun log2_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(0.5f, 1f, 2f, 4f, 8f)) { _, x -> x.log2() }
    }

    @Test
    fun log10_backward_matches_finite_diff() {
        assertGradMatchesFiniteDiff(floatArrayOf(1f, 10f, 100f)) { _, x -> x.log10() }
    }
}
