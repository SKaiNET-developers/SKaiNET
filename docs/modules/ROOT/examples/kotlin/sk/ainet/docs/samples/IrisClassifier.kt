package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.data.iris.Iris
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32
import kotlin.random.Random
// end::imports[]

/**
 * The classifier the Android getting-started tutorial builds: a `[4, 16, 3]` MLP
 * trained on the embedded Iris dataset with cross-entropy over one-hot targets.
 *
 * Every API used here is `commonMain` and available on the Android target; this
 * module compiles and runs it in CI on the JVM so the tutorial cannot rot.
 */
object IrisClassifier {

    data class Result(val firstLoss: Float, val lastLoss: Float, val accuracy: Float)

    suspend fun run(): Result {
        // tag::data[]
        // 150 rows, embedded in skainet-data-simple — no download, works on every target.
        // Features [n, 4], targets one-hot [n, 3]; stratified split keeps class balance.
        val (trainSet, testSet) = Iris.load().split(0.8, seed = 42L, stratified = true)
        val train = trainSet.dataBatch<FP32, Float>(0, trainSet.size)
        val test = testSet.dataBatch<FP32, Float>(0, testSet.size)
        // end::data[]

        // tag::model[]
        // A graph (autograd) context for training; a plain CPU context for inference.
        val baseCtx = DirectCpuExecutionContext()
        val trainCtx = DefaultGraphExecutionContext(
            baseOps = baseCtx.ops,
            phase = Phase.TRAIN,
            createTapeFactory = { _ -> DefaultGradientTape() },
        )

        val rng = Random(42)
        val model = sequential<FP32, Float>(trainCtx) {
            input(4)                                              // sepal/petal measurements
            dense(16) { weights { randn(std = 0.5f, random = rng) } }
            activation { it.relu() }
            dense(3) { weights { randn(std = 0.5f, random = rng) } } // class logits
        }
        // end::model[]

        // tag::train[]
        val x = train.x[0]
        val y = train.y
        val runner = training<FP32, Float> {
            model { model }
            loss { CrossEntropyLoss() } // applies softmax internally — the model outputs logits
            optimizer {
                sgd(lr = 0.05).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        var firstLoss = 0f
        var lastLoss = 0f
        repeat(300) { epoch ->
            val loss = runner.step(trainCtx, x, y).data.get()
            if (epoch == 0) firstLoss = loss
            lastLoss = loss
        }
        // end::train[]

        // tag::evaluate[]
        // Held-out accuracy on a fresh inference context: argmax of the logits.
        val evalCtx = DirectCpuExecutionContext()
        val preds = model.forward(test.x[0], evalCtx)
        val n = testSet.size
        var correct = 0
        for (i in 0 until n) {
            var best = 0
            var bestScore = preds.data.get(i, 0)
            var target = 0
            var targetScore = test.y.data.get(i, 0)
            for (c in 1 until 3) {
                val s = preds.data.get(i, c)
                if (s > bestScore) { best = c; bestScore = s }
                val t = test.y.data.get(i, c)
                if (t > targetScore) { target = c; targetScore = t }
            }
            if (best == target) correct++
        }
        val accuracy = correct.toFloat() / n
        // end::evaluate[]

        return Result(firstLoss, lastLoss, accuracy)
    }
}
