package sk.ainet.compile.c

import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.matmul
import sk.ainet.context.data
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SinusApproximatorTapeTest {

    @Test
    fun testSinusApproximatorRecording() {
        val model = SinusApproximator()
        val ctx = DefaultGraphExecutionContext.tape()
        
        println("[DEBUG_LOG] Starting recording...")
        val (tape, _) = ctx.record {
            val currentTape = this.currentTape!!
            val globalStack = sk.ainet.tape.Execution.tapeStack
            globalStack.pushTape(currentTape)
            
            try {
                val module = model.create(this)
                
                val input = data<FP32, Float>(this) {
                    tensor {
                        shape(1, 1) {
                            fromArray(floatArrayOf(0.0f))
                        }
                    }
                }
                
                println("[DEBUG_LOG] Running forward pass...")
                module.forward(input, this)
                println("[DEBUG_LOG] Forward pass finished.")
            } finally {
                globalStack.popTape()
            }
        }
        
        assertNotNull(tape, "Tape should not be null")
        assertTrue(tape is DefaultExecutionTape, "Tape should be DefaultExecutionTape")
        
        // Let's see how many operations were recorded
        // Since we don't have direct access to recorded operations list in ExecutionTape interface,
        // we might need to cast to DefaultExecutionTape or use toComputeGraph()
        
        val graph = (tape as DefaultExecutionTape).toComputeGraph()
        println("[DEBUG_LOG] Graph nodes: ${graph.nodes.size}")
        graph.nodes.forEach { node ->
            println("[DEBUG_LOG] Node ID: ${node.id}, Op: ${node.operationName}, Inputs: ${node.inputs.map { it.name }}, Outputs: ${node.outputs.map { it.name }}")
        }
        
        // Check if we have MatMul and Add operations
        val opNames = graph.nodes.map { it.operationName }
        
        if (graph.nodes.size < 8) {
            println("[DEBUG_LOG] Recorded operations on tape: ${(tape as DefaultExecutionTape).operations.size}")
            (tape as DefaultExecutionTape).operations.forEach { op ->
                println("[DEBUG_LOG] Recorded Op: ${op.operation::class.simpleName}")
            }
        }
        
        assertTrue(graph.nodes.size >= 8, "Graph should have at least 8 nodes, but has ${graph.nodes.size}")
    }

    @Test
    fun testSimpleManualRecording() {
        val ctx = DefaultGraphExecutionContext.tape()
        val (tape, _) = ctx.record {
            val a = full<FP32, Float>(Shape(1, 1), FP32::class, 1.0f)
            val b = full<FP32, Float>(Shape(1, 1), FP32::class, 2.0f)
            println("[DEBUG_LOG] Performing add...")
            val c = a + b
            println("[DEBUG_LOG] Performing matmul...")
            val d = c.matmul(b)
            d
        }
        
        val graph = (tape as DefaultExecutionTape).toComputeGraph()
        println("[DEBUG_LOG] Manual Graph nodes: ${graph.nodes.size}")
        graph.nodes.forEach { node ->
            println("[DEBUG_LOG] Manual Node: ${node.id}, Op: ${node.operation::class.simpleName}")
        }
        assertTrue(graph.nodes.size >= 2, "Should have at least 2 op nodes")
    }
}
