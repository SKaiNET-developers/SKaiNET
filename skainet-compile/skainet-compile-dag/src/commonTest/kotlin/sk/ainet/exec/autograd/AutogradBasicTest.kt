package sk.ainet.exec.autograd

import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.utils.drawDot
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.*
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertTrue

class AutogradBasicTest {

    private fun createTrainCtx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        val graph = sk.ainet.lang.graph.DefaultComputeGraph()
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
            computeGraph = graph,
            baseSink = GraphSink(graph)
        )
    }

    @Test
    fun showcase_autograd_with_cosine_distance() {
        val ctx = createTrainCtx()

        // 1. Initialize data
        // We'll have two vectors. 'a' will be a trainable parameter.
        val aTensor = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 0f, 0f)).withRequiresGrad()
        val bTensor = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(0f, 1f, 0f))
        
        // Wrap 'a' as a parameter for the optimizer
        val aParam = ModuleParameter.WeightParameter("a", aTensor, true)

        // 2. Forward pass with recording (Conversion to graph)
        println("--- Forward Pass ---")
        val pair = ctx.record {
            // Cosine distance: 1 - (a dot b) / (||a|| * ||b||)
            aTensor.cosineDistance(bTensor)
        }
        val tape = pair.first as DefaultGradientTape
        val distance = pair.second
        
        val initialDistance = distance.data.get()
        println("Initial cosine distance: $initialDistance")

        // 3. Export untrained graph for visualization
        val forwardGraph = ctx.computeGraph!!
        val forwardDot = drawDot(forwardGraph)
        println("--- Forward Graph (DOT) ---")
        println(forwardDot.content)
        assertTrue(forwardDot.content.contains("->"), "Graph should contain operations")

        // 4. Add loss function
        // For this example, we want to minimize the distance, so the distance itself is our loss.
        val loss = distance

        // 5. Graph inversion (Backward pass)
        println("--- Backward Pass ---")
        tape.computeGradients(targets = listOf(loss), sources = listOf(aTensor))

        assertTrue(aTensor.grad != null, "Gradients should be computed for 'a'")
        // Gradient of Cosine Distance w.r.t 'a':
        // d(1 - similarity)/da = - d(similarity)/da
        println("Gradient of a: ${aTensor.grad!!.data.get(0)}, ${aTensor.grad!!.data.get(1)}, ${aTensor.grad!!.data.get(2)}")

        // 6. Export inverted graph for visualization
        // In this version of Skainet, the tape records both forward and backward ops if computeGradients is called?
        // Actually, toComputeGraph() usually shows the forward trace recorded.
        val backwardGraph = tape.toComputeGraph()
        val backwardDot = drawDot(backwardGraph)
        println("--- Backward Graph (DOT) ---")
        println(backwardDot.content)

        // 7. Optimization step
        println("--- Optimization Step ---")
        val optimizer = sgd(lr = 0.5)
        optimizer.addParameter(aParam)
        
        optimizer.step()
        optimizer.zeroGrad()

        // Verify distance decreased
        val pairFinal = ctx.record {
            aTensor.cosineDistance(bTensor)
        }
        val finalDistance = pairFinal.second
        val distanceAfterStep = finalDistance.data.get()
        println("Distance after optimization step: $distanceAfterStep")
        
        assertTrue(distanceAfterStep < initialDistance + 1e-6f, "Distance should decrease or stay same after optimization step")
        println("Success: Distance changed from $initialDistance to $distanceAfterStep")
    }
}
