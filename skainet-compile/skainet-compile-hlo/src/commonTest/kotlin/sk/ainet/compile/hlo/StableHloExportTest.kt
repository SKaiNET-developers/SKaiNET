package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.types.DType
import kotlin.collections.emptyList

class StableHloExportTest {

    @Test
    fun testSimpleStableHloExport() {
        // Build a tiny graph: input a, input b -> add -> relu
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(1, 4), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(1, 4), "FP32"))
        )
        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(1, 4), "FP32"),
                TensorSpec("b", listOf(1, 4), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(1, 4), "FP32"))
        )
        val relu = GraphNode(
            id = "relu1",
            operation = ReluOperation<DType, Any>(),
            inputs = listOf(TensorSpec("c", listOf(1, 4), "FP32")),
            outputs = listOf(TensorSpec("d", listOf(1, 4), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)
        graph.addNode(relu)

        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))
        graph.addEdge(GraphEdge("e3", add, relu, 0, 0, add.outputs[0]))

        val mlir = toStableHlo(graph, "main").content

        println("[DEBUG_LOG] StableHLO export:\n$mlir")

        assertTrue(mlir.contains("module {"))
        assertTrue(mlir.contains("func.func @main"))
        assertTrue(mlir.contains("stablehlo.add"))
        assertTrue(mlir.contains("stablehlo.maximum"))
    }
}
