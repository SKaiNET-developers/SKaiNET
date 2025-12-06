package sk.ainet.compile.graph

import sk.ainet.lang.dag.dag
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GraphProgramCompilerTest {

    @Test
    fun dagDsl_toComputeGraph_buildsNodesEdgesAndValidates() {
        val program = dag {
            val x = input("x", TensorSpec("x", listOf(1, 4), "FP32"))
            val w = parameter("w", TensorSpec("w", listOf(4, 4), "FP32"))
            val mm = matmul(x, w)
            val y = relu(mm)
            output(y)
        }

        val graph = program.toComputeGraph() as DefaultComputeGraph

        // Structure checks
        assertEquals(4, graph.nodes.size, "input, param, matmul, relu expected")
        assertEquals(3, graph.edges.size, "Edges from input/param to matmul and matmul to relu expected")

        // Topological order should include all nodes and finish at relu
        val topo = graph.getTopologicalOrder()
        assertEquals(4, topo.size)
        assertEquals("relu", topo.last().operation.name.lowercase())

        // Validation must succeed
        val validation = graph.validate()
        assertTrue(validation is sk.ainet.lang.tensor.ops.ValidationResult.Valid, "Graph should validate: $validation")
    }
}
