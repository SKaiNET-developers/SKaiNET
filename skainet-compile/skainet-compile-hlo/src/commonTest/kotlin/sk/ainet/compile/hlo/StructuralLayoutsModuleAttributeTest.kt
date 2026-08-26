package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.withBlockOrder
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * #1179: the module header carries *structure*, not display names. `skainet.tensor_layouts` holds
 * the machine-readable facts — block element counts, block bytes, bit widths, block order — that a
 * downstream consumer needs to size and address packed weights; `kind` carries the name, so the
 * former `skainet.tensor_encodings` names dictionary (which nothing outside this repo ever read)
 * is gone rather than duplicated. `TurboQuantPolar(4b, 128)` used to degrade to a name whose block
 * size was unrecoverable; now it is two integers in the header.
 */
class StructuralLayoutsModuleAttributeTest {

    @Test
    fun structural_facts_reach_the_module_header() {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(1, 4), "FP32")),
        )
        // A feed-ordered Q4_K weight: the block order must ride along as a string.
        val q4Spec = TensorSpec("w_q4", listOf(1, 4), "FP32")
            .withTensorEncoding(TensorEncoding.Q4_K)
            .withBlockOrder("INPUT_BLOCK_MAJOR")
        val q4Node = GraphNode(
            id = "w_q4",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(q4Spec),
        )
        // A TurboQuant weight: bits and block size must be recoverable from the header.
        val tqSpec = TensorSpec("w_tq", listOf(1, 4), "FP32")
            .withTensorEncoding(TensorEncoding.TurboQuantPolar(bitsPerElement = 4, blockSize = 128))
        val tqNode = GraphNode(
            id = "w_tq",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(tqSpec),
        )
        val add1 = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(inputA.outputs[0], q4Spec),
            outputs = listOf(TensorSpec("sum1", listOf(1, 4), "FP32")),
        )
        val add2 = GraphNode(
            id = "add2",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(add1.outputs[0], tqSpec),
            outputs = listOf(TensorSpec("sum2", listOf(1, 4), "FP32")),
        )

        graph.addNode(inputA); graph.addNode(q4Node); graph.addNode(tqNode)
        graph.addNode(add1); graph.addNode(add2)
        graph.addEdge(GraphEdge("e1", inputA, add1, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", q4Node, add1, 0, 1, q4Spec))
        graph.addEdge(GraphEdge("e3", add1, add2, 0, 0, add1.outputs[0]))
        graph.addEdge(GraphEdge("e4", tqNode, add2, 0, 1, tqSpec))

        val mlir = toStableHlo(graph, "structural_chain").content

        assertTrue(mlir.contains("skainet.tensor_layouts"), "structural dictionary must be emitted:\n$mlir")
        assertTrue(
            mlir.contains("w_q4 = {kind = \"Q4_K\", block_elems = 256, block_bytes = 144, block_order = \"INPUT_BLOCK_MAJOR\"}"),
            "Q4_K facts + block order must be machine-readable:\n$mlir",
        )
        assertTrue(
            mlir.contains("w_tq = {kind = \"TurboQuant-Polar-4b\", bits = 4, block_elems = 128}")
                || mlir.contains("bits = 4, block_elems = 128"),
            "TurboQuant bits and block size must be recoverable:\n$mlir",
        )
        // the redundant names dictionary is gone, not duplicated
        assertTrue(!mlir.contains("skainet.tensor_encodings"), "legacy names dictionary must not be emitted:\n$mlir")
    }

    @Test
    fun dense_graph_keeps_bare_module_header() {
        // A graph with no encoding metadata must emit the bare `module {` header with no
        // `attributes` block. A null tensorEncoding is the unknown / not-carried state — not
        // Dense — and the emitter must stay silent.
        val graph = DefaultComputeGraph()
        val a = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(1, 4), "FP32")),
        )
        val b = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(1, 4), "FP32")),
        )
        val add = GraphNode(
            id = "add",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(a.outputs[0], b.outputs[0]),
            outputs = listOf(TensorSpec("sum", listOf(1, 4), "FP32")),
        )
        graph.addNode(a); graph.addNode(b); graph.addNode(add)
        graph.addEdge(GraphEdge("e1", a, add, 0, 0, a.outputs[0]))
        graph.addEdge(GraphEdge("e2", b, add, 0, 1, b.outputs[0]))

        val mlir = toStableHlo(graph, "dense_chain").content
        assertTrue(mlir.contains("module {"), "bare module header expected:\n$mlir")
        assertTrue(!mlir.contains("module attributes"), "no attributes block for a dense graph:\n$mlir")
    }
}
