package sk.ainet.apps.sine

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
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

fun main(args: Array<String>) {
    val epochs = 2000
    val batchSize= 64
    val lr = 0.05
    
    println("Starting Sine Approximation Training...")
    println("Epochs: $epochs, Batch Size: $batchSize, Learning Rate: $lr")

    val baseCtx = DirectCpuExecutionContext()
    // Ensure Phase.TRAIN is set for training context and use DefaultGradientTape
    val trainCtxFinal = DefaultGraphExecutionContext(
        baseOps = baseCtx.ops,
        phase = Phase.TRAIN,
        createTapeFactory = { _ -> DefaultGradientTape(true) }
    )

    // 1. Define Model - using trainCtxFinal to ensure parameters are registered
    val model = sequential<FP32, Float>(trainCtxFinal) {
        input(1)
        dense(16) {
            weights { randn(std = 0.5f) }
        }
        activation { it.relu() }
        dense(16) {
            weights { randn(std = 0.5f) }
        }
        activation { it.relu() }
        dense(1) {
            weights { randn(std = 0.5f) }
        }
    }

    // 2. Prepare Data
    val xValues = FloatArray(batchSize) { i ->
        (i.toFloat() / (batchSize - 1)) * (PI.toFloat() / 2f)
    }
    val yValues = FloatArray(batchSize) { i -> sin(xValues[i].toDouble()).toFloat() }

    // Use trainCtxFinal directly for data creation to ensure session registration
    val inputs = trainCtxFinal.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(batchSize, 1), FP32::class, xValues)
    val targets = trainCtxFinal.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(batchSize, 1), FP32::class, yValues)

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
    println("Training...")
    val dataset = listOf(inputs to targets)
    repeat(epochs) { epoch ->
        var totalLoss = 0f
        for ((x, y) in dataset) {
            val lossTensor = runner.step(trainCtxFinal, x, y)
            totalLoss += (lossTensor.data.get() as Float)
        }
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            println("Epoch ${epoch + 1}/$epochs, Loss: ${totalLoss / dataset.size}")
        }
    }

    // 6. Evaluation
    println("\nEvaluation:")
    val evalCtx = DefaultGraphExecutionContext(baseOps = baseCtx.ops, phase = Phase.EVAL)
    val testPoints = floatArrayOf(0.0f, PI.toFloat() / 4f, PI.toFloat() / 2f)
    for (tp in testPoints) {
        val inputTensor = baseCtx.fromFloatArray<FP32, Float>(sk.ainet.lang.tensor.Shape(1, 1), FP32::class, floatArrayOf(tp))
        val prediction = model.forward(inputTensor, evalCtx)
        val predVal = prediction.data[0, 0]
        val actualVal = sin(tp.toDouble()).toFloat()
        println("x: ${"%.4f".format(tp)}, Predicted sin(x): ${"%.4f".format(predVal)}, Actual sin(x): ${"%.4f".format(actualVal)}, Error: ${"%.4f".format(kotlin.math.abs(predVal - actualVal))}")
    }
}
