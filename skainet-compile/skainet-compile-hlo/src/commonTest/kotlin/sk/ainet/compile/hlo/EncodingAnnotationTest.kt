package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Covers the emitter hook added for #473: when a [TensorSpec] flowing
 * through the graph carries a non-null `tensorEncoding`, the emitted
 * StableHLO module must preserve that information as a comment so
 * downstream tools (and humans reading the MLIR) can see that
 * quantization flowed through the compile boundary.
 */
class EncodingAnnotationTest {

    @Test
    fun q8_0_weight_produces_tensor_encoding_comment() {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(1, 4), "FP32"))
        )

        // A Q8_0 weight, synthesized the way TraceToGraphBuilder.finalize()
        // produces its weight nodes after #469 landed: the spec carries
        // TensorEncoding.Q8_0 on its metadata.
        val weightSpec = TensorSpec("w", listOf(1, 4), "FP32")
            .withTensorEncoding(TensorEncoding.Q8_0)
        val weightNode = GraphNode(
            id = "w",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(weightSpec)
        )

        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(1, 4), "FP32"),
                weightSpec
            ),
            outputs = listOf(TensorSpec("out", listOf(1, 4), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(weightNode)
        graph.addNode(add)
        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", weightNode, add, 0, 1, weightSpec))

        val mlir = toStableHlo(graph, "quant_add").content
        println("[DEBUG_LOG] quant-annotated export:\n$mlir")

        // The emitter must have surfaced the encoding as a comment near
        // the weight input's initialization. MLIR tools ignore comments
        // but the text round-trips preserve them, so this is the cheapest
        // way to keep SKaiNET's quantization metadata visible through the
        // StableHLO emit boundary.
        assertTrue(
            mlir.contains("tensor_encoding"),
            "emitter must include a tensor_encoding annotation comment"
        )
        assertTrue(
            mlir.contains("encoding=Q8_0"),
            "annotation must name the concrete TensorEncoding (Q8_0)"
        )
        assertTrue(
            mlir.contains("name=w"),
            "annotation must identify the tensor the encoding applies to"
        )
    }

    @Test
    fun dense_graph_emits_no_encoding_comment() {
        // An all-FP32 graph with no encoding metadata must not introduce
        // spurious tensor_encoding comments. A `null` tensorEncoding is
        // the unknown / not-carried state, not "Dense", and the emitter
        // must treat it as silent.
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

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)
        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))

        val mlir = toStableHlo(graph, "dense_add").content
        assertFalse(
            mlir.contains("tensor_encoding"),
            "dense graph must not emit any tensor_encoding annotation"
        )
    }
}
