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
 * Covers the structured module-level attribute emission for #477:
 * every TensorSpec flowing through the graph with a non-null
 * tensorEncoding must appear in a single `skainet.tensor_encodings`
 * dictionary on the emitted `module attributes { ... }` header, so
 * downstream tools can read it with one attribute lookup instead of
 * string-matching against scattered comments.
 */
class TensorEncodingsModuleAttributeTest {

    @Test
    fun encoded_weights_produce_module_attributes_block() {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(1, 4), "FP32"))
        )

        // Two weight inputs with distinct encodings, exactly the shape
        // TraceToGraphBuilder.finalize() produces post-#469 when a session
        // resolves quantized weights.
        val q8Spec = TensorSpec("w_q8", listOf(1, 4), "FP32")
            .withTensorEncoding(TensorEncoding.Q8_0)
        val q8Node = GraphNode(
            id = "w_q8",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(q8Spec)
        )

        val q4Spec = TensorSpec("w_q4", listOf(1, 4), "FP32")
            .withTensorEncoding(TensorEncoding.Q4_K)
        val q4Node = GraphNode(
            id = "w_q4",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(q4Spec)
        )

        val add1 = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(TensorSpec("a", listOf(1, 4), "FP32"), q8Spec),
            outputs = listOf(TensorSpec("sum1", listOf(1, 4), "FP32"))
        )
        val add2 = GraphNode(
            id = "add2",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(TensorSpec("sum1", listOf(1, 4), "FP32"), q4Spec),
            outputs = listOf(TensorSpec("sum2", listOf(1, 4), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(q8Node)
        graph.addNode(q4Node)
        graph.addNode(add1)
        graph.addNode(add2)
        graph.addEdge(GraphEdge("e1", inputA, add1, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", q8Node, add1, 0, 1, q8Spec))
        graph.addEdge(GraphEdge("e3", add1, add2, 0, 0, add1.outputs[0]))
        graph.addEdge(GraphEdge("e4", q4Node, add2, 0, 1, q4Spec))

        val mlir = toStableHlo(graph, "quant_chain").content
        println("[DEBUG_LOG] module-attribute export:\n$mlir")

        // The emitted module header must carry a structured attribute
        // enumerating every encoded tensor in one place.
        assertTrue(
            mlir.contains("module attributes"),
            "module header must be emitted with `module attributes { ... }` when encodings are present"
        )
        assertTrue(
            mlir.contains("skainet.tensor_encodings"),
            "module attributes must include the `skainet.tensor_encodings` dictionary"
        )

        // Both encoded tensors must appear in the dictionary by name,
        // each mapped to its TensorEncoding.name.
        assertTrue(
            mlir.contains("w_q8 = \"Q8_0\""),
            "dictionary must map `w_q8` to `\"Q8_0\"`"
        )
        assertTrue(
            mlir.contains("w_q4 = \"Q4_K\""),
            "dictionary must map `w_q4` to `\"Q4_K\"`"
        )
    }

    @Test
    fun dense_graph_keeps_bare_module_header() {
        // A graph with no encoding metadata must emit the bare
        // `module {` header with no `attributes` block. A `null`
        // tensorEncoding is the unknown / not-carried state — not
        // Dense — and the emitter must stay silent.
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
            mlir.contains("module attributes"),
            "dense graph must not emit a `module attributes` block"
        )
        assertFalse(
            mlir.contains("skainet.tensor_encodings"),
            "dense graph must not emit the `skainet.tensor_encodings` dictionary"
        )
        assertTrue(
            mlir.contains("module {"),
            "dense graph must keep the bare `module {` header"
        )
    }
}
