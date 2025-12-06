package sk.ainet.compile.graph

import sk.ainet.lang.dag.dag
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Basic sanity test: build a tiny graph for y = relu(x + bias) via the DAG DSL.
 */
class SimpleExpressionGraphTest {

    @Test
    fun simple_add_relu_graph_builds_and_validates() {
        val program = dag {
            val x = input("x", TensorSpec("x", listOf(1), "FP32"))
            val bias = constant<FP32, Float>("bias") { shape(1) { zeros() } }
            val sum = add(x, bias)
            val y = relu(sum)
            output(y)
        }

        val graph = program.toComputeGraph() as DefaultComputeGraph

        // input + const + add + relu
        assertEquals(4, graph.nodes.size)
        assertEquals(3, graph.edges.size)

        val ops = graph.nodes.map { it.operation.name }.toSet()
        assertTrue("add" in ops)
        assertTrue("relu" in ops)

        val validation = graph.validate()
        assertTrue(validation is sk.ainet.lang.tensor.ops.ValidationResult.Valid, "Graph must validate: $validation")

        // Only relu should be an output node
        val outputNodes = graph.getOutputNodes()
        assertEquals(1, outputNodes.size)
        assertTrue(outputNodes.first().operation.name.equals("relu", ignoreCase = true))
    }
}
