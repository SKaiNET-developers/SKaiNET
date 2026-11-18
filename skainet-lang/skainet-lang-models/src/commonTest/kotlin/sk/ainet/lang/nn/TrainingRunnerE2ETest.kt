package sk.ainet.lang.nn

import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TrainingRunnerE2ETest {

    private fun createTrainCtx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) }
        )
    }

    @Test
    fun test_trainStep_e2e() {
        val ctx = createTrainCtx()

        // 1. Setup simple model: y = w * x
        val wInitial = 10f
        val wTensor = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, wInitial)
        wTensor.gradState.requiresGrad = true
        val w = ModuleParameter.WeightParameter<FP32, Float>("w", wTensor, true)

        val model = object : Module<FP32, Float>() {
            override val name: String = "simple"
            override val modules: List<Module<FP32, Float>> = emptyList()
            override val params: List<ModuleParameter<FP32, Float>> = listOf(w)
            override fun forward(input: Tensor<FP32, Float>, ctx: sk.ainet.context.ExecutionContext): Tensor<FP32, Float> {
                val boundInput = input.bind(ctx)
                val boundW = w.value.bind(ctx)
                val result = ctx.ops.matmul(boundInput, boundW) as Tensor<FP32, Float>
                if (ctx is DefaultGraphExecutionContext) {
                    println("[DEBUG_LOG] forward1: result.id=${ctx.session.refOf(result).id}")
                }
                return result
            }
        }

        val loss = MSELoss()
        val optimizer = sgd(lr = 0.1)
        optimizer.addParameter(w)

        val x = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0f)
        val y = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 5.0f) // Target is 5

        // 2. Run trainStep
        val initialLossValue = trainStep(model, loss, optimizer, ctx, x, y)
        val initialLoss = initialLossValue.data.get() as Float
        println("[DEBUG_LOG] after first trainStep: w.value.grad=${w.value.grad}")
        
        val secondLossValue = trainStep(model, loss, optimizer, ctx, x, y)
        val secondLoss = secondLossValue.data.get() as Float
        println("[DEBUG_LOG] after second trainStep: w.value.grad=${w.value.grad}")

        // 3. Verify loss decreased
        println("Initial loss: $initialLoss, Second loss: $secondLoss")
        assertTrue(secondLoss < initialLoss, "Loss should decrease after training step")
        assertTrue(w.value.data[0, 0] < wInitial, "Weight should have moved towards target")
    }

    @Test
    fun test_training_dsl_e2e() {
        val ctx = createTrainCtx()

        // 1. Setup simple model
        val wTensor = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 10f)
        wTensor.gradState.requiresGrad = true
        val w = ModuleParameter.WeightParameter<FP32, Float>("w", wTensor, true)

        val myModel = object : Module<FP32, Float>() {
            override val name: String = "simple"
            override val modules: List<Module<FP32, Float>> = emptyList()
            override val params: List<ModuleParameter<FP32, Float>> = listOf(w)
            override fun forward(input: Tensor<FP32, Float>, ctx: sk.ainet.context.ExecutionContext): Tensor<FP32, Float> {
                val boundW = w.value.bind(ctx)
                val result = ctx.ops.matmul(input, boundW) as Tensor<FP32, Float>
                println("[DEBUG_LOG] forward2: w.value=${w.value}, boundW=$boundW, w.value.ops=${w.value.ops}, boundW.ops=${boundW.ops}, ctx.ops=${ctx.ops}")
                return result
            }
        }

        // 2. Use DSL
        val runner = training<FP32, Float> {
            model { myModel }
            loss { MSELoss() }
            optimizer { 
                sgd(lr = 0.1).apply {
                    addParameter(w)
                }
            }
        }

        val x = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0f)
        val y = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 5.0f)

        // 3. Run training loop using DSL runner
        val initialLoss = runner.step(ctx, x, y).data.get() as Float
        
        // Use the train helper for 5 epochs
        val dataset = listOf(x to y)
        runner.train(ctx, dataset, epochs = 5)
        
        val finalLoss = runner.step(ctx, x, y).data.get() as Float
        
        println("DSL - Initial loss: $initialLoss, Final loss: $finalLoss")
        assertTrue(finalLoss < initialLoss, "Loss should decrease after training loop")
    }

    /**
     * #1527: `trainStep` never cleared the trace session, so every tensor of every step stayed
     * reachable and a training loop ran out of memory.
     */
    @Test
    fun test_training_loop_does_not_accumulate_session_tensors() {
        val ctx = createTrainCtx()

        val wTensor = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 10f)
        wTensor.gradState.requiresGrad = true
        val w = ModuleParameter.WeightParameter<FP32, Float>("w", wTensor, true)

        val myModel = object : Module<FP32, Float>() {
            override val name: String = "simple"
            override val modules: List<Module<FP32, Float>> = emptyList()
            override val params: List<ModuleParameter<FP32, Float>> = listOf(w)
            override fun forward(input: Tensor<FP32, Float>, ctx: sk.ainet.context.ExecutionContext): Tensor<FP32, Float> =
                ctx.ops.matmul(input, w.value.bind(ctx)) as Tensor<FP32, Float>
        }

        val runner = training<FP32, Float> {
            model { myModel }
            loss { MSELoss() }
            optimizer { sgd(lr = 0.01).apply { addParameter(w) } }
        }

        val x = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0f)
        val y = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 5.0f)

        val first = runner.step(ctx, x, y).data.get() as Float
        repeat(9) { runner.step(ctx, x, y) }
        val afterTen = ctx.session.trackedTensorCount
        var last = first
        repeat(190) { last = runner.step(ctx, x, y).data.get() as Float }
        val afterTwoHundred = ctx.session.trackedTensorCount

        assertEquals(afterTen, afterTwoHundred, "the session must not grow from step 10 to step 200")
        assertEquals(0, afterTwoHundred, "a finished step leaves no tensor references behind")
        assertTrue(last < first, "training still converges with the session cleared each step ($first -> $last)")
    }
}
