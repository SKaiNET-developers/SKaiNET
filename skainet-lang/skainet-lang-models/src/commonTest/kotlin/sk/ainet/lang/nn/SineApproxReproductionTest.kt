package sk.ainet.lang.nn

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class SineApproxReproductionTest {

    @Test
    fun testSineApproximationTraining() {
        // 50 epochs should be enough for this simple sine approx and safer for JS timeouts
        val epochs = 50 
        val batchSize = 64
        val lr = 0.01
        val seed = 42

        val baseCtx = DirectCpuExecutionContext()
        val trainCtxFinal = DefaultGraphExecutionContext(
            baseOps = baseCtx.ops,
            phase = Phase.TRAIN,
            createTapeFactory = { _ -> DefaultGradientTape() }
        )

        val random = Random(seed)

        // 1. Define Model
        val model = sequential<FP32, Float>(trainCtxFinal) {
            input(1)
            dense(16) {
                weights { randn(std = 0.1f, random = random) }
            }
            activation { it.relu() }
            dense(16) {
                weights { randn(std = 0.1f, random = random) }
            }
            activation { it.relu() }
            dense(1) {
                weights { randn(std = 0.1f, random = random) }
            }
        }

        // 2. Prepare Data
        val xValues = FloatArray(batchSize) { i ->
            (i.toFloat() / (batchSize - 1)) * (PI.toFloat() / 2f)
        }
        val yValues = FloatArray(batchSize) { i -> sin(xValues[i].toDouble()).toFloat() }

        val inputs = baseCtx.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(batchSize, 1), FP32::class, xValues)
        val targets = baseCtx.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(batchSize, 1), FP32::class, yValues)

        // 3. Configure Training
        val runner = training<FP32, Float> {
            model { model }
            loss { MSELoss() }
            optimizer {
                sgd(lr = lr).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        // 5. Training Loop
        val dataset = listOf(inputs to targets)
        var firstLoss = 0f
        var lastLoss = 0f

        repeat(epochs) { epoch ->
            var totalLoss = 0f
            for ((x, y) in dataset) {
                val lossTensor = runner.step(trainCtxFinal, x, y)
                totalLoss += (lossTensor.data.get() as Float)
            }
            val avgLoss = totalLoss / dataset.size
            if (epoch == 0) firstLoss = avgLoss
            lastLoss = avgLoss
            
            if ((epoch + 1) % 10 == 0) {
                println("[DEBUG_LOG] Epoch ${epoch + 1}/$epochs, Loss: $avgLoss")
            }
        }

        println("[DEBUG_LOG] First Loss: $firstLoss, Last Loss: $lastLoss")
        assertTrue(firstLoss > 0.0001f, "First loss should be positive, but was $firstLoss")
        assertTrue(lastLoss < firstLoss, "Loss should decrease from $firstLoss to $lastLoss")
    }
}
