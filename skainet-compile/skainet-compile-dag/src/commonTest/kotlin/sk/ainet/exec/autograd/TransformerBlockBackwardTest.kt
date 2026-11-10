package sk.ainet.exec.autograd

import kotlin.math.abs
import kotlin.random.Random
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

/**
 * A whole pre-norm Transformer block trains through the tape: token and position embeddings (`gather`), RMSNorm from
 * primitives, two-head self-attention (`scaledDotProductAttention`, causal), a gated GELU feed-forward, a class-token
 * head and a weighted-sum loss. Every parameter's analytic gradient is checked against central finite differences,
 * which is the end-to-end claim behind #1384: with the attention adjoint, nothing in a Transformer block stops the
 * gradient.
 */
class TransformerBlockBackwardTest {

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

    private fun randn(n: Int, seed: Int, scale: Float): FloatArray = Random(seed).let { r -> FloatArray(n) { (r.nextFloat() * 2f - 1f) * scale } }

    // ── the block ──────────────────────────────────────────────────────────────────────────────

    private companion object {
        const val VOCAB = 7
        const val N = 2       // sequences
        const val L = 4       // tokens per sequence
        const val DIM = 4
        const val HEADS = 2
        const val HEAD_DIM = DIM / HEADS
        const val FFN = 6
        const val CLASSES = 3
        val TOKENS = floatArrayOf(1f, 3f, 5f, 2f, 6f, 0f, 4f, 1f)   // [N, L], used as gather indices

        /** Parameter name → shape, in a fixed order. */
        val PARAMS: List<Pair<String, Shape>> = listOf(
            "tok" to Shape(VOCAB, DIM), "pos" to Shape(L, DIM),
            "norm1" to Shape(DIM), "norm2" to Shape(DIM), "normOut" to Shape(DIM),
            "wq" to Shape(DIM, DIM), "wk" to Shape(DIM, DIM), "wv" to Shape(DIM, DIM), "wo" to Shape(DIM, DIM),
            "wGate" to Shape(DIM, FFN), "wUp" to Shape(DIM, FFN), "wDown" to Shape(FFN, DIM),
            "head" to Shape(DIM, CLASSES),
        )
    }

    private fun initialValues(): List<FloatArray> = PARAMS.mapIndexed { i, (name, shape) ->
        when {
            name.startsWith("norm") -> FloatArray(shape.volume) { 1f + 0.1f * (it % 3) }
            else -> randn(shape.volume, 100 + i, 0.6f)
        }
    }

    private fun rmsNorm(c: DefaultGraphExecutionContext, x: Tensor<FP32, Float>, g: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val ops = c.ops
        val last = x.rank - 1
        val ms = ops.mean(ops.multiply(x, x), last)
        val inv = ops.unsqueeze(ops.rdivScalar(1.0, ops.sqrt(ops.addScalar(ms, 1e-5))), last)
        return ops.multiply(ops.multiply(x, inv), g)
    }

    private fun heads(c: DefaultGraphExecutionContext, x: Tensor<FP32, Float>): Tensor<FP32, Float> =
        c.ops.permute(c.ops.reshape(x, Shape(N, L, HEADS, HEAD_DIM)), intArrayOf(0, 2, 1, 3))

    /** The loss: `sum(logits ⊙ w)` with fixed weights, so every logit matters. */
    @Suppress("UNCHECKED_CAST")
    private fun loss(c: DefaultGraphExecutionContext, p: Map<String, Tensor<FP32, Float>>): Tensor<FP32, Float> {
        val ops = c.ops
        val tokens = c.floats(Shape(N, L), TOKENS) as Tensor<DType, *>
        val positions = c.floats(Shape(L), FloatArray(L) { it.toFloat() }) as Tensor<DType, *>
        val x = ops.add(ops.gather(p.getValue("tok"), tokens, 0), ops.gather(p.getValue("pos"), positions, 0))   // [N, L, DIM]
        val h = ops.reshape(rmsNorm(c, x, p.getValue("norm1")), Shape(N * L, DIM))
        val q = heads(c, ops.matmul(h, p.getValue("wq")))
        val k = heads(c, ops.matmul(h, p.getValue("wk")))
        val v = heads(c, ops.matmul(h, p.getValue("wv")))
        val att = ops.scaledDotProductAttention(q, k, v, causal = true)
        val merged = ops.reshape(ops.permute(att, intArrayOf(0, 2, 1, 3)), Shape(N * L, DIM))
        val x1 = ops.add(x, ops.reshape(ops.matmul(merged, p.getValue("wo")), Shape(N, L, DIM)))
        val h2 = ops.reshape(rmsNorm(c, x1, p.getValue("norm2")), Shape(N * L, DIM))
        val f = ops.matmul(ops.multiply(ops.gelu(ops.matmul(h2, p.getValue("wGate"))), ops.matmul(h2, p.getValue("wUp"))), p.getValue("wDown"))
        val x2 = ops.add(x1, ops.reshape(f, Shape(N, L, DIM)))
        val cls = ops.reshape(ops.narrow(x2, 1, L - 1, 1), Shape(N, DIM))   // last token sees everything under the causal mask
        val logits = ops.matmul(rmsNorm(c, cls, p.getValue("normOut")), p.getValue("head"))
        val w = c.floats(Shape(N, CLASSES), FloatArray(N * CLASSES) { 0.5f + 0.25f * it })
        val weighted = ops.multiply(logits, w)
        return weighted.ops.sum(weighted)
    }

    private fun lossAt(values: List<FloatArray>): Float {
        val c = ctx()
        val p = PARAMS.mapIndexed { i, (name, shape) -> name to c.floats(shape, values[i]) }.toMap()
        return buf(loss(c, p))[0]
    }

    @Test
    fun every_parameter_of_a_transformer_block_matches_finite_differences() {
        val values = initialValues()
        val c = ctx()
        val params = PARAMS.mapIndexed { i, (name, shape) -> name to c.floats(shape, values[i].copyOf()).withRequiresGrad() }
        val (tape, l) = c.record { loss(this as DefaultGraphExecutionContext, params.toMap()) }
        (tape as DefaultGradientTape).computeGradients(targets = listOf(l), sources = params.map { it.second })

        val eps = 1e-2f
        val tol = 2e-2f
        var checked = 0
        var worst = 0f
        for ((i, named) in params.withIndex()) {
            val (name, t) = named
            val grad = buf(assertNotNull(t.grad, "$name has no gradient"))
            for (j in grad.indices) {
                val plus = values.mapIndexed { n, a -> if (n == i) a.copyOf().also { it[j] += eps } else a }
                val minus = values.mapIndexed { n, a -> if (n == i) a.copyOf().also { it[j] -= eps } else a }
                val fd = (lossAt(plus) - lossAt(minus)) / (2 * eps)
                val diff = abs(grad[j] - fd)
                worst = maxOf(worst, diff)
                assertTrue(diff <= tol, "$name[$j]: analytic=${grad[j]} finite-diff=$fd diff=$diff tol=$tol")
                checked++
            }
        }
        assertTrue(checked == PARAMS.sumOf { it.second.volume }, "checked $checked entries")
        println("transformer block: $checked parameter entries within $tol of finite differences (worst $worst)")
    }
}
