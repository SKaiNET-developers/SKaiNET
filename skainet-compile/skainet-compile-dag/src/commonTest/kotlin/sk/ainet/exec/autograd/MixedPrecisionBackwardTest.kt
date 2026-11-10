package sk.ainet.exec.autograd

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.collectParams
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * SKEEP-008 phase A3 (#1433): `convert` is differentiable between floating-point dtypes, so
 * gradients cross a [sk.ainet.lang.nn.PrecisionBoundary] and the layers behind it train.
 */
class MixedPrecisionBackwardTest {

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

    private fun values(t: Tensor<*, *>): FloatArray {
        val dims = t.shape.dimensions
        val idx = IntArray(dims.size)
        return FloatArray(t.shape.volume) {
            val v = (t.data.get(*idx) as Number).toFloat()
            for (d in dims.indices.reversed()) {
                if (++idx[d] < dims[d]) break
                idx[d] = 0
            }
            v
        }
    }

    /** `sum(out ⊙ weights)`: the gradient of this loss w.r.t. `out` is `weights`. */
    private fun weightedSum(c: DefaultGraphExecutionContext, out: Tensor<FP32, Float>, weights: FloatArray): Tensor<FP32, Float> {
        val weighted = out.ops.multiply(out, c.floats(out.shape, weights))
        return weighted.ops.sum(weighted)
    }

    private fun gradOf(
        c: DefaultGraphExecutionContext,
        x: Tensor<FP32, Float>,
        loss: DefaultGraphExecutionContext.() -> Tensor<FP32, Float>,
    ): FloatArray {
        val (tape, l) = c.record { (this as DefaultGraphExecutionContext).loss() }
        (tape as DefaultGradientTape).computeGradients(targets = listOf(l), sources = listOf(x))
        return values(assertNotNull(x.grad, "no gradient reached the source"))
    }

    // ── the cast itself ───────────────────────────────────────────────────────────────────────

    @Test
    fun convert_roundTripThroughFp16_passesTheGradientStraightThrough() {
        val c = ctx()
        val x = c.floats(Shape(2, 2), floatArrayOf(0.1234567f, -1.7654321f, 2.3456789f, 0.0123456f)).withRequiresGrad()
        val w = floatArrayOf(1f, 2f, 3f, 4f) // exactly representable in FP16
        val g = gradOf(c, x) {
            weightedSum(this, ops.convert(ops.convert(x, FP16), FP32), w)
        }
        assertContentEquals(w, g)
    }

    @Test
    fun convert_gradientEnteringANarrowRegion_isRoundedToThatFormat() {
        val c = ctx()
        val x = c.floats(Shape(1, 2), floatArrayOf(0.5f, -0.25f)).withRequiresGrad()
        val w = floatArrayOf(0.1234567f, 1.7654321f) // not representable in BF16
        val g = gradOf(c, x) {
            weightedSum(this, ops.convert(ops.convert(x, BF16), FP32), w)
        }
        for (i in w.indices) {
            assertTrue(abs(g[i] - w[i]) <= 1e-2f * abs(w[i]), "gradient $i drifted: ${g[i]} vs ${w[i]}")
        }
        assertTrue(g.indices.any { g[it] != w[it] }, "the gradient crossed a BF16 tensor and must show its rounding")
    }

    @Test
    fun convert_toTheSameDtype_doesNotDoubleTheGradient() {
        // The CPU backend returns its input here, yet the call is recorded: input and output
        // of the recorded op are one tensor, and the rule must not add its gradient to itself.
        val c = ctx()
        val x = c.floats(Shape(1, 3), floatArrayOf(1f, 2f, 3f)).withRequiresGrad()
        val w = floatArrayOf(5f, 6f, 7f)
        val g = gradOf(c, x) { weightedSum(this, ops.convert(x, FP32), w) }
        assertContentEquals(w, g)
    }

    @Test
    fun convert_throughAnIntegerDtype_hasNoGradient() {
        val c = ctx()
        val x = c.floats(Shape(1, 3), floatArrayOf(1.5f, 2.5f, 3.5f)).withRequiresGrad()
        val g = gradOf(c, x) {
            weightedSum(this, ops.convert(ops.convert(x, Int32), FP32), floatArrayOf(5f, 6f, 7f))
        }
        assertContentEquals(floatArrayOf(0f, 0f, 0f), g)
    }

    // ── through the DSL boundary ──────────────────────────────────────────────────────────────

    private val w1 = floatArrayOf(
        0.1234567f, -1.7654321f,
        2.3456789f, 0.0123456f,
        -0.9876543f, 1.1111111f,
    )
    private val b1 = floatArrayOf(0.1111111f, -0.2222222f, 0.3333333f)
    private val w2 = floatArrayOf(
        1.9999999f, -0.1111111f, 0.5555555f,
        -1.3333333f, 0.7777777f, -0.4444444f,
    )
    private val b2 = floatArrayOf(0.0101010f, -0.0202020f)
    private val lossWeights = floatArrayOf(1f, -2f)
    private val x0 = floatArrayOf(0.3f, -1.1f)

    /** input(2) → dense(3) → dense(2), linear, with the first layer in [first] and the second in a stage of [second]. */
    private fun <A : DType, B : DType> network(c: DefaultGraphExecutionContext, first: A, second: B): Module<FP32, Float> =
        definition {
            network<FP32, Float>(c) {
                // The Input module sets the flag on the tensor it is given; without it the
                // first cast has nothing to propagate and the input gets no gradient.
                input(2, requiresGrad = true)
                dense(3, dtype = first, id = "layer1") {
                    weights { fromArray(w1) }
                    bias { fromArray(b1) }
                }
                stage("head", dtype = second) {
                    dense(2, "layer2") {
                        weights { fromArray(w2) }
                        bias { fromArray(b2) }
                    }
                }
            }
        }

    private fun <A : DType, B : DType> lossAt(first: A, second: B, input: FloatArray): Float {
        val c = ctx()
        val model = network(c, first, second)
        val out = model.forward(c.floats(Shape(1, 2), input), c)
        return values(weightedSum(c, out, lossWeights))[0]
    }

    private class Grads(val input: FloatArray, val params: Map<String, FloatArray>, val paramDtypes: Map<String, Any>)

    private fun <A : DType, B : DType> grads(first: A, second: B): Grads {
        val c = ctx()
        val model = network(c, first, second)
        val x = c.floats(Shape(1, 2), x0).withRequiresGrad()
        val params = model.collectParams()
        assertEquals(4, params.size, "two dense layers behind boundaries: ${params.map { it.name }}")
        val (tape, loss) = c.record { weightedSum(c, model.forward(x, c), lossWeights) }
        @Suppress("UNCHECKED_CAST")
        val sources = (listOf(x) + params.map { it.value }) as List<Tensor<FP32, Float>>
        (tape as DefaultGradientTape).computeGradients(targets = listOf(loss), sources = sources)
        return Grads(
            input = values(assertNotNull(x.grad)),
            params = params.associate { p -> p.name to values(assertNotNull(p.value.grad, "${p.name} has no gradient")) },
            paramDtypes = params.associate { p -> p.name to assertNotNull(p.value.grad).dtype },
        )
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, relTol: Float, what: String) {
        assertEquals(expected.size, actual.size, what)
        val scale = expected.maxOf { abs(it) }.coerceAtLeast(1f)
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= relTol * scale,
                "$what[$i]: expected ${expected[i]}, got ${actual[i]} (tolerance ${relTol * scale})",
            )
        }
    }

    @Test
    fun gradientCheck_throughFp16Boundaries_matchesFiniteDifferences() {
        // The network is linear, so a central difference is exact up to the rounding the
        // boundaries add; a wide step keeps that rounding small against the difference.
        val h = 0.5f
        val numeric = FloatArray(x0.size) { i ->
            val plus = x0.copyOf().also { it[i] += h }
            val minus = x0.copyOf().also { it[i] -= h }
            (lossAt(FP16, FP16, plus) - lossAt(FP16, FP16, minus)) / (2 * h)
        }
        assertClose(numeric, grads(FP16, FP16).input, relTol = 1e-2f, what = "dLoss/dInput")
    }

    @Test
    fun gradients_throughBf16DenseAndFp16Stage_matchTheFp32Network() {
        val reference = grads(FP32, FP32)
        val mixed = grads(BF16, FP16)

        assertClose(reference.input, mixed.input, relTol = 3e-2f, what = "dLoss/dInput")
        for ((name, expected) in reference.params) {
            assertClose(expected, assertNotNull(mixed.params[name]), relTol = 3e-2f, what = "dLoss/d$name")
        }
        assertTrue(
            reference.params.any { (name, expected) -> !expected.contentEquals(mixed.params[name]) },
            "narrow gradients must show the rounding, or the casts never happened",
        )
        // A parameter's gradient has the parameter's dtype, so an optimizer updates like with like.
        assertTrue(mixed.paramDtypes.values.containsAll(listOf(BF16::class, FP16::class)), "${mixed.paramDtypes}")
    }

    @Test
    fun training_updatesTheLayersBehindABoundary_andLowersTheLoss() {
        val c = ctx()
        val model = network(c, BF16, FP16)
        val x = c.floats(Shape(1, 2), x0)
        val target = c.floats(Shape(1, 2), floatArrayOf(0.5f, -0.5f))
        val params = model.collectParams()
        val before = params.associate { it.name to values(it.value) }

        fun loss(): Tensor<FP32, Float> {
            val out = model.forward(x, c)
            val diff = out.ops.subtract(out, target)
            val sq = diff.ops.multiply(diff, diff)
            return sq.ops.sum(sq)
        }

        val initial = values(loss())[0]
        val optimizer = sgd(lr = 0.01)
        @Suppress("UNCHECKED_CAST")
        params.forEach { optimizer.addParameter(it as ModuleParameter<FP32, Float>) }
        repeat(5) {
            val (tape, l) = c.record { loss() }
            @Suppress("UNCHECKED_CAST")
            (tape as DefaultGradientTape).computeGradients(
                targets = listOf(l),
                sources = params.map { it.value } as List<Tensor<FP32, Float>>,
            )
            optimizer.step()
            optimizer.zeroGrad()
        }
        val final = values(loss())[0]

        assertTrue(final < initial, "loss must go down: $initial -> $final")
        for (p in params) {
            assertTrue(
                !before.getValue(p.name).contentEquals(values(p.value)),
                "${p.name} sits behind a boundary and must have been updated",
            )
        }
    }
}
