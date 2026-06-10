package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.tanh
import sk.ainet.lang.types.FP32
import kotlin.random.Random
// end::imports[]

/**
 * End-to-end training with the SKaiNET `training { }` DSL: a tiny two-cluster
 * classification task learned by an MLP `[2, 8, 1]` with `tanh` activations.
 *
 * Demonstrates the pieces issue #102 asks to document — running a model, a loss,
 * an optimizer, and a metric (classification accuracy) — as real, CI-run code.
 */
object TrainingDemo {

    /** firstLoss/lastLoss show learning; accuracy is the held-out metric. */
    data class Result(val firstLoss: Float, val lastLoss: Float, val accuracy: Float)

    // Two linearly separable clusters: label +1 near (1,1), label -1 near (-1,-1).
    private val featuresFlat = floatArrayOf(
        1.0f, 1.1f, 0.9f, 1.2f, 1.2f, 0.8f, 1.1f, 0.9f,
        -1.0f, -1.1f, -0.9f, -1.2f, -1.2f, -0.8f, -1.1f, -0.9f,
    )
    private val labelsFlat = floatArrayOf(1f, 1f, 1f, 1f, -1f, -1f, -1f, -1f)

    fun run(): Result {
        val n = labelsFlat.size

        // tag::setup[]
        // A graph (autograd) context for training; a plain CPU context for inference.
        val baseCtx = DirectCpuExecutionContext()
        val trainCtx = DefaultGraphExecutionContext(
            baseOps = baseCtx.ops,
            phase = Phase.TRAIN,
            createTapeFactory = { _ -> DefaultGradientTape() },
        )

        val rng = Random(42)
        val model = sequential<FP32, Float>(trainCtx) {
            input(2)
            dense(8) { weights { randn(std = 0.5f, random = rng) } }
            activation { it.tanh() }
            dense(1) { weights { randn(std = 0.5f, random = rng) } }
            activation { it.tanh() }
        }

        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(n, 2), FP32::class, featuresFlat)
        val y = baseCtx.fromFloatArray<FP32, Float>(Shape(n, 1), FP32::class, labelsFlat)
        // end::setup[]

        // tag::loop[]
        val runner = training<FP32, Float> {
            model { model }
            loss { MSELoss() }
            optimizer {
                sgd(lr = 0.1).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        var firstLoss = 0f
        var lastLoss = 0f
        repeat(150) { epoch ->
            val loss = runner.step(trainCtx, x, y).data.get()
            if (epoch == 0) firstLoss = loss
            lastLoss = loss
        }
        // end::loop[]

        // tag::accuracy[]
        // Metric: classification accuracy on a fresh inference context.
        val evalCtx = DirectCpuExecutionContext()
        val preds = model.forward(x, evalCtx)
        var correct = 0
        for (i in 0 until n) {
            val score = preds.data.get(i, 0)
            val predicted = if (score >= 0f) 1f else -1f
            if (predicted == labelsFlat[i]) correct++
        }
        val accuracy = correct.toFloat() / n
        // end::accuracy[]

        return Result(firstLoss, lastLoss, accuracy)
    }
}
