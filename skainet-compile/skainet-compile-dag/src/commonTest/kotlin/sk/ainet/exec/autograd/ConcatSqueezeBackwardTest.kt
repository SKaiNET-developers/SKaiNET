package sk.ainet.exec.autograd

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
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
import sk.ainet.lang.types.FP32

/**
 * `concatBackward` and `squeezeBackward` on the shapes that used to go wrong silently: inputs of
 * different sizes along the concat axis, and `squeeze` with no axis. The loss is a weighted sum,
 * so each position's gradient is the weight that sat at that position — an exact expectation.
 */
class ConcatSqueezeBackwardTest {

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

    /** Gradients of [inputs] for `sum(f(inputs) ⊙ weights)`, where the weights count up from 1 in row-major order. */
    private fun grads(
        c: DefaultGraphExecutionContext,
        inputs: List<Tensor<FP32, Float>>,
        f: DefaultGraphExecutionContext.(List<Tensor<FP32, Float>>) -> Tensor<FP32, Float>,
    ): List<Tensor<FP32, Float>> {
        val (tape, loss) = c.record {
            val out = f(this as DefaultGraphExecutionContext, inputs)
            val w = (this as DefaultGraphExecutionContext).floats(out.shape, FloatArray(out.shape.volume) { (it + 1).toFloat() })
            val weighted = out.ops.multiply(out, w)
            weighted.ops.sum(weighted)
        }
        (tape as DefaultGradientTape).computeGradients(targets = listOf(loss), sources = inputs)
        return inputs.mapIndexed { i, x -> assertNotNull(x.grad, "input $i has no gradient") }
    }

    private fun assertGrad(expectedShape: Shape, expected: FloatArray, actual: Tensor<FP32, Float>) {
        assertEquals(expectedShape.dimensions.toList(), actual.shape.dimensions.toList(), "gradient shape")
        assertContentEquals(expected, buf(actual))
    }

    // ── concat ────────────────────────────────────────────────────────────────────────────────

    @Test
    fun concat_unequal_sizes_along_dim_1() {
        val c = ctx()
        val a = c.floats(Shape(2, 1), floatArrayOf(1f, 2f)).withRequiresGrad()
        val b = c.floats(Shape(2, 2), floatArrayOf(3f, 4f, 5f, 6f)).withRequiresGrad()
        // out = [[a0, b0, b1], [a1, b2, b3]] with weights 1..6
        val g = grads(c, listOf(a, b)) { (x, y) -> ops.concat(listOf(x, y), 1) }
        assertGrad(Shape(2, 1), floatArrayOf(1f, 4f), g[0])
        assertGrad(Shape(2, 2), floatArrayOf(2f, 3f, 5f, 6f), g[1])
    }

    @Test
    fun concat_three_inputs_of_different_sizes_along_dim_0() {
        val c = ctx()
        val a = c.floats(Shape(1, 2), floatArrayOf(1f, 2f)).withRequiresGrad()
        val b = c.floats(Shape(3, 2), FloatArray(6) { it.toFloat() }).withRequiresGrad()
        val d = c.floats(Shape(2, 2), FloatArray(4) { it.toFloat() }).withRequiresGrad()
        // out rows: a (1), b (3), d (2) → 6 rows, weights 1..12
        val g = grads(c, listOf(a, b, d)) { (x, y, z) -> ops.concat(listOf(x, y, z), 0) }
        assertGrad(Shape(1, 2), floatArrayOf(1f, 2f), g[0])
        assertGrad(Shape(3, 2), floatArrayOf(3f, 4f, 5f, 6f, 7f, 8f), g[1])
        assertGrad(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f), g[2])
    }

    @Test
    fun concat_equal_sizes_still_split_correctly() {
        val c = ctx()
        val a = c.floats(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)).withRequiresGrad()
        val b = c.floats(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)).withRequiresGrad()
        val g = grads(c, listOf(a, b)) { (x, y) -> ops.concat(listOf(x, y), 1) }
        assertGrad(Shape(2, 2), floatArrayOf(1f, 2f, 5f, 6f), g[0])
        assertGrad(Shape(2, 2), floatArrayOf(3f, 4f, 7f, 8f), g[1])
    }

    @Test
    fun concat_negative_dim() {
        val c = ctx()
        val a = c.floats(Shape(2, 1), floatArrayOf(1f, 2f)).withRequiresGrad()
        val b = c.floats(Shape(2, 2), floatArrayOf(3f, 4f, 5f, 6f)).withRequiresGrad()
        val g = grads(c, listOf(a, b)) { (x, y) -> ops.concat(listOf(x, y), -1) }
        assertGrad(Shape(2, 1), floatArrayOf(1f, 4f), g[0])
        assertGrad(Shape(2, 2), floatArrayOf(2f, 3f, 5f, 6f), g[1])
    }

    // ── squeeze ───────────────────────────────────────────────────────────────────────────────

    @Test
    fun squeeze_all_axes_restores_the_input_shape() {
        val c = ctx()
        val x = c.floats(Shape(1, 3, 1), floatArrayOf(1f, 2f, 3f)).withRequiresGrad()
        val g = grads(c, listOf(x)) { (v) -> ops.squeeze(v) }
        assertGrad(Shape(1, 3, 1), floatArrayOf(1f, 2f, 3f), g[0])
    }

    @Test
    fun squeeze_inner_axis_restores_the_input_shape() {
        val c = ctx()
        val x = c.floats(Shape(2, 1, 2), floatArrayOf(1f, 2f, 3f, 4f)).withRequiresGrad()
        val g = grads(c, listOf(x)) { (v) -> ops.squeeze(v, 1) }
        assertGrad(Shape(2, 1, 2), floatArrayOf(1f, 2f, 3f, 4f), g[0])
    }

    @Test
    fun squeeze_then_concat_chain() {
        val c = ctx()
        val x = c.floats(Shape(1, 2), floatArrayOf(1f, 2f)).withRequiresGrad()
        val y = c.floats(Shape(3), floatArrayOf(3f, 4f, 5f)).withRequiresGrad()
        // squeeze(x) → [2], concat with y → [5], weights 1..5
        val g = grads(c, listOf(x, y)) { (a, b) -> ops.concat(listOf(ops.squeeze(a), b), 0) }
        assertGrad(Shape(1, 2), floatArrayOf(1f, 2f), g[0])
        assertGrad(Shape(3), floatArrayOf(3f, 4f, 5f), g[1])
    }
}
