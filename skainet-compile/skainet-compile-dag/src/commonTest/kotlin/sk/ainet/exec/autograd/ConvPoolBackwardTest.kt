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
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.FP32

/**
 * Tier C parity tests for conv1d/2d/3d and pool/upsample backward formulas.
 * Compares analytic gradient vs central finite difference on a small, fixed
 * tensor — small enough to keep the O(window*kernel) brute-force backward
 * cheap. FP32 tolerance is generous (3e-2) because conv accumulates many
 * products and FP32 rounding adds up.
 */
class ConvPoolBackwardTest {

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

    private fun buf(t: Tensor<*, *>): FloatArray = (t.data as FloatArrayTensorData<*>).buffer

    /**
     * Compares analytic gradient w.r.t. `x` against central finite-difference
     * for `f`. `f` builds a scalar (sum-reduced) output from `x` plus any
     * captured constants. Output is sum-reduced inside the recording scope
     * so the gradient corresponds to df/dx element-wise.
     */
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
            assertTrue(
                diff <= tol,
                "[$i] analytic=${analytic[i]} fd=$fdGrad diff=$diff tol=$tol",
            )
        }
    }

    private fun FloatArray.sumElems(): Float {
        var s = 0f
        for (v in this) s += v
        return s
    }

    @Test
    fun conv2d_backward_input_matches_finite_diff() {
        // Input [1, 1, 4, 4], weight [2, 1, 2, 2], stride 1, padding 0.
        val w = floatArrayOf(
            1f, 0f, 0f, -1f,    // out-channel 0
            0.5f, -0.5f, 1f, 1f, // out-channel 1
        )
        val bShape = Shape(2)
        val bias = floatArrayOf(0.1f, -0.2f)
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 4, 4),
            x0 = floatArrayOf(
                0.5f, 1f, -0.3f, 0.8f,
                -1f, 0.2f, 0.7f, -0.4f,
                0.1f, -0.9f, 0.6f, 0.3f,
                0.4f, 0.5f, -0.2f, 1f,
            ),
        ) { c, x ->
            val wT = floatTensor(c, Shape(2, 1, 2, 2), w)
            val bT = floatTensor(c, bShape, bias)
            x.ops.conv2d(x, wT, bT, stride = 1 to 1, padding = 0 to 0, dilation = 1 to 1, groups = 1)
        }
    }

    @Test
    fun conv2d_backward_input_with_stride_and_padding() {
        val w = floatArrayOf(0.3f, -0.1f, 0.8f, 0.2f)
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 5, 5),
            x0 = FloatArray(25) { (it % 7 - 3).toFloat() * 0.2f },
        ) { c, x ->
            val wT = floatTensor(c, Shape(1, 1, 2, 2), w)
            x.ops.conv2d(x, wT, null, stride = 2 to 2, padding = 1 to 1, dilation = 1 to 1, groups = 1)
        }
    }

    @Test
    fun conv1d_backward_input_matches_finite_diff() {
        val w = floatArrayOf(0.5f, -1f, 0.2f, 1f, 0.3f, -0.4f)
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 2, 6),
            x0 = FloatArray(12) { (it - 6) * 0.15f },
        ) { c, x ->
            // weight [C_out=1, C_in=2, kL=3]
            val wT = floatTensor(c, Shape(1, 2, 3), w)
            x.ops.conv1d(x, wT, null, stride = 1, padding = 0, dilation = 1, groups = 1)
        }
    }

    @Test
    fun conv3d_backward_input_matches_finite_diff() {
        val w = floatArrayOf(
            0.5f, -0.3f, 1f, 0.2f,
            -1f, 0.4f, 0.1f, 0.7f,
        )
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 3, 3, 3),
            x0 = FloatArray(27) { (it % 5 - 2) * 0.1f },
        ) { c, x ->
            val wT = floatTensor(c, Shape(1, 1, 2, 2, 2), w)
            x.ops.conv3d(
                x, wT, null,
                stride = Triple(1, 1, 1),
                padding = Triple(0, 0, 0),
                dilation = Triple(1, 1, 1),
                groups = 1,
            )
        }
    }

    @Test
    fun maxPool2d_backward_routes_to_argmax() {
        // Distinct values per window — no ties.
        val x0 = floatArrayOf(
            1f, 5f, 2f, 6f,
            3f, 7f, 4f, 8f,
            9f, 13f, 10f, 14f,
            11f, 15f, 12f, 16f,
        )
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 4, 4),
            x0 = x0,
            eps = 1e-2f, // larger eps — argmax must not jump under perturbation
        ) { _, x ->
            x.ops.maxPool2d(x, kernelSize = 2 to 2, stride = 2 to 2, padding = 0 to 0)
        }
    }

    @Test
    fun avgPool2d_backward_distributes_uniformly() {
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 4, 4),
            x0 = FloatArray(16) { it.toFloat() * 0.1f - 0.7f },
        ) { _, x ->
            x.ops.avgPool2d(
                x,
                kernelSize = 2 to 2, stride = 2 to 2, padding = 0 to 0,
                countIncludePad = true,
            )
        }
    }

    @Test
    fun split_backward_accumulates_chunk_grads() {
        // split a length-6 vector into three length-2 chunks, multiply each
        // by a distinct scalar, sum the lot. Each input element's gradient
        // should equal the scalar of the chunk it belongs to (2, 3, 5).
        assertGradMatchesFiniteDiff(
            xShape = Shape(6),
            x0 = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f),
        ) { _, x ->
            val chunks = x.ops.split(x, splitSize = 2, dim = 0)
            val a = x.ops.mulScalar(chunks[0], 2f)
            val b = x.ops.mulScalar(chunks[1], 3f)
            val c = x.ops.mulScalar(chunks[2], 5f)
            x.ops.add(x.ops.add(a, b), c)
        }
    }

    @Test
    fun upsample2d_nearest_backward_sums_block() {
        assertGradMatchesFiniteDiff(
            xShape = Shape(1, 1, 3, 3),
            x0 = FloatArray(9) { (it - 4) * 0.25f },
        ) { _, x ->
            x.ops.upsample2d(x, scale = 2 to 2, mode = UpsampleMode.Nearest, alignCorners = false)
        }
    }
}
