package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.LLMFusionPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LLMFusionPassTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(1, 4096)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun opNode(id: String, opName: String, params: Map<String, Any> = emptyMap()) = GraphNode(
        id = id,
        operation = GenericOperation(opName, parameters = params),
        inputs = listOf(spec()),
        outputs = listOf(spec())
    )

    // ---- QKV Merge Tests ----

    @Test
    fun fusesQKVProjectionsWithNamedNodes() {
        // Source node → 3 matmul nodes named q_proj, k_proj, v_proj
        val graph = DefaultComputeGraph()
        val norm = graph.addNode(opNode("norm", "rms_norm"))
        val q = graph.addNode(opNode("attn_q_proj", "matmul"))
        val k = graph.addNode(opNode("attn_k_proj", "matmul"))
        val v = graph.addNode(opNode("attn_v_proj", "matmul"))

        graph.addEdge(GraphEdge("e1", norm, q, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", norm, k, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", norm, v, tensorSpec = spec()))

        val result = LLMFusionPass().apply(graph)
        assertTrue(result.changed)
        assertTrue(result.graph.nodes.any { it.operation.name == "fused_qkv_proj" })
    }

    @Test
    fun fusesThreeAnonymousMatmulsAsQKV() {
        // Source node → exactly 3 matmul nodes (fallback heuristic)
        val graph = DefaultComputeGraph()
        val norm = graph.addNode(opNode("norm", "rms_norm"))
        val m1 = graph.addNode(opNode("m1", "matmul"))
        val m2 = graph.addNode(opNode("m2", "matmul"))
        val m3 = graph.addNode(opNode("m3", "matmul"))

        graph.addEdge(GraphEdge("e1", norm, m1, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", norm, m2, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", norm, m3, tensorSpec = spec()))

        val result = LLMFusionPass().apply(graph)
        assertTrue(result.changed)
        assertTrue(result.graph.nodes.any { it.operation.name == "fused_qkv_proj" })
    }

    @Test
    fun doesNotFuseOnlyTwoMatmuls() {
        val graph = DefaultComputeGraph()
        val norm = graph.addNode(opNode("norm", "rms_norm"))
        val m1 = graph.addNode(opNode("m1", "matmul"))
        val m2 = graph.addNode(opNode("m2", "matmul"))

        graph.addEdge(GraphEdge("e1", norm, m1, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", norm, m2, tensorSpec = spec()))

        val result = LLMFusionPass().apply(graph)
        // Only 2 matmuls: doesn't match QKV pattern
        assertFalse(result.graph.nodes.any { it.operation.name == "fused_qkv_proj" })
    }

    // ---- SwiGLU Fusion Tests ----

    @Test
    fun fusesSwiGluFFN() {
        // gate_matmul → silu → multiply(with up_matmul) → down_matmul
        // Both gate and up share the same input (norm output)
        val graph = DefaultComputeGraph()
        val norm = graph.addNode(opNode("norm", "rms_norm"))
        val gate = graph.addNode(opNode("gate", "matmul"))
        val up = graph.addNode(opNode("up", "matmul"))
        val silu = graph.addNode(opNode("silu", "silu"))
        val mul = graph.addNode(opNode("mul", "multiply"))
        val down = graph.addNode(opNode("down", "matmul"))

        // norm → gate → silu → mul → down
        graph.addEdge(GraphEdge("e1", norm, gate, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", norm, up, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", gate, silu, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e4", silu, mul, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e5", up, mul, destinationInputIndex = 1, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e6", mul, down, tensorSpec = spec()))

        val result = LLMFusionPass().apply(graph)
        assertTrue(result.changed)
        assertTrue(result.graph.nodes.any { it.operation.name == "fused_swiglu_ffn" })
    }

    // ---- General Tests ----

    @Test
    fun emptyGraphUnchanged() {
        val result = LLMFusionPass().apply(DefaultComputeGraph())
        assertFalse(result.changed)
    }

    @Test
    fun diagnosticsReportFusions() {
        val graph = DefaultComputeGraph()
        val norm = graph.addNode(opNode("norm", "rms_norm"))
        val q = graph.addNode(opNode("attn_q_proj", "matmul"))
        val k = graph.addNode(opNode("attn_k_proj", "matmul"))
        val v = graph.addNode(opNode("attn_v_proj", "matmul"))

        graph.addEdge(GraphEdge("e1", norm, q, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", norm, k, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", norm, v, tensorSpec = spec()))

        val result = LLMFusionPass().apply(graph)
        assertTrue(result.diagnostics.isNotEmpty())
        assertTrue(result.diagnostics.any { "QKV" in it || "qkv" in it })
    }
}
