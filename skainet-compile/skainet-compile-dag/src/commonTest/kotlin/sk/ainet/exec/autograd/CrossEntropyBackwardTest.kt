package sk.ainet.exec.autograd

import kotlin.math.abs
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.loss.Reduction
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Regression for issue #862: CrossEntropyLoss must backpropagate to the
 * predictions. The index-target path used to build its result host-side (via
 * tensorDataFactory + fromData), which detached the tape and produced null /
 * zero gradients — training silently froze. The soft-target path only recorded
 * when the targets lived on the recording context.
 */
class CrossEntropyBackwardTest {

    private fun graphCtx(): DefaultGraphExecutionContext {
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

    private fun buf(t: Tensor<*, *>): FloatArray = (t.data as FloatArrayTensorData<*>).buffer

    private fun softmaxRow(logits: FloatArray): FloatArray {
        val m = logits.max()
        val ex = logits.map { exp((it - m).toDouble()) }
        val s = ex.sum()
        return FloatArray(logits.size) { (ex[it] / s).toFloat() }
    }

    @Test
    fun index_target_cross_entropy_backprops_to_predictions() {
        val c = graphCtx()
        val predsFlat = floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f) // [2,3]
        val preds = c.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, predsFlat).withRequiresGrad()
        val targets = c.fromIntArray<Int32, Int>(Shape(2), Int32::class, intArrayOf(2, 0))

        val pair = c.record {
            CrossEntropyLoss().forward(preds, targets, this, Reduction.MEAN)
        }
        val loss = pair.second
        (pair.first as DefaultGradientTape).computeGradients(targets = listOf(loss), sources = listOf(preds))

        val grad = preds.grad
        assertNotNull(grad, "CrossEntropyLoss must populate preds.grad (was detached in #862)")

        // Analytic mean-CE gradient: (softmax(row) - oneHot) / N
        val n = 2
        val expected = FloatArray(6)
        val cls = intArrayOf(2, 0)
        for (row in 0 until 2) {
            val sm = softmaxRow(predsFlat.copyOfRange(row * 3, row * 3 + 3))
            for (j in 0 until 3) {
                expected[row * 3 + j] = (sm[j] - if (j == cls[row]) 1f else 0f) / n
            }
        }
        val actual = buf(grad)
        for (i in expected.indices) {
            assertTrue(abs(actual[i] - expected[i]) < 1e-4f, "[$i] expected ${expected[i]} got ${actual[i]}")
        }
    }

    @Test
    fun soft_target_cross_entropy_records_even_with_eager_targets() {
        val c = graphCtx()
        val predsFlat = floatArrayOf(1f, 3f, 2f, 0f) // [2,2]
        val preds = c.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, predsFlat).withRequiresGrad()

        // Targets created on a *separate, eager* context — the soft-target multiply
        // must still record through the predictions' ops.
        val eager = DirectCpuExecutionContext()
        val targets = eager.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(0.25f, 0.75f, 0.6f, 0.4f))

        val pair = c.record {
            CrossEntropyLoss().forward(preds, targets, this, Reduction.MEAN)
        }
        (pair.first as DefaultGradientTape).computeGradients(targets = listOf(pair.second), sources = listOf(preds))

        val grad = preds.grad
        assertNotNull(grad, "soft-target CrossEntropyLoss must populate preds.grad")
        // Gradient must be non-trivial (not all zeros)
        assertTrue(buf(grad).any { abs(it) > 1e-6f }, "gradient should be non-zero")
    }
}
