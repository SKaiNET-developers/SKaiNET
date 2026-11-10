package sk.ainet.compile.opt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.opt.passes.DTypeConstraintResolutionPass
import sk.ainet.compile.opt.passes.DtypeConstraintViolationException
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

class DTypeConstraintResolutionPassTest {

    private fun node(
        id: String,
        opName: String = "matmul",
        inputDtype: String = "Float32",
        policy: DTypePolicy? = null,
    ): GraphNode {
        val meta = if (policy != null) mapOf<String, Any>(DTypeConstraintResolutionPass.POLICY_KEY to policy) else emptyMap()
        return GraphNode(
            id = id,
            operation = GenericOperation(opName),
            inputs = listOf(TensorSpec(name = "$id-in", shape = listOf(4, 4), dtype = inputDtype)),
            outputs = listOf(TensorSpec(name = "$id-out", shape = listOf(4, 4), dtype = inputDtype)),
            metadata = meta,
        )
    }

    @Test
    fun nodes_without_policy_are_passed_through() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0"))
        g.addNode(node("n1"))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertFalse(result.changed, "no policy = no work")
        // Neither node should be marked resolved (only visited nodes get the marker).
        assertEquals(emptyList(), result.graph.nodes.filter { it.metadata.containsKey(DTypeConstraintResolutionPass.RESOLVED_KEY) })
    }

    @Test
    fun any_policy_passes_through() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", policy = DTypePolicy.Any))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertTrue(result.changed, "the resolved-marker write counts as a change")
        val n = result.graph.nodes.single()
        assertTrue(n.metadata[DTypeConstraintResolutionPass.RESOLVED_KEY] == true)
    }

    @Test
    fun require_matching_dtype_passes() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "Float32", policy = DTypePolicy.Require(FP32)))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertTrue(result.changed)
    }

    @Test
    fun require_mismatched_dtype_fails_fast() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "Float32", policy = DTypePolicy.Require(BF16)))
        val ex = assertFailsWith<DtypeConstraintViolationException> {
            DTypeConstraintResolutionPass().apply(g)
        }
        val msg = ex.message ?: ""
        assertTrue(msg.contains("BFloat16"), "msg must name the required dtype: $msg")
        assertTrue(msg.contains("Float32"), "msg must name the actual input dtype: $msg")
        assertTrue(msg.contains("Cast kernels"), "msg must hint at the resolution path: $msg")
    }

    @Test
    fun require_mismatched_dtype_with_short_alias_also_resolves() {
        // DAG DSL emits dtype strings like "FP32" / "BF16" via dtypeName().
        // The pass must handle both the registry canonical name and the short alias.
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "FP32", policy = DTypePolicy.Require(FP32)))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertTrue(result.changed, "alias 'FP32' must satisfy Require(FP32)")
    }

    @Test
    fun prefer_mismatched_dtype_emits_diagnostic_no_throw() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "Float32", policy = DTypePolicy.Prefer(BF16)))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertTrue(result.changed)
        assertTrue(
            result.diagnostics.any { it.contains("prefers") && it.contains("BFloat16") },
            "diagnostic must mention the preference: ${result.diagnostics}",
        )
    }

    @Test
    fun oneOf_in_set_passes() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "Float32", policy = DTypePolicy.OneOf(setOf(FP32, BF16))))
        val result = DTypeConstraintResolutionPass().apply(g)
        assertTrue(result.changed)
    }

    @Test
    fun oneOf_outside_set_fails_fast() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", inputDtype = "Float32", policy = DTypePolicy.OneOf(setOf(BF16, Int8))))
        val ex = assertFailsWith<DtypeConstraintViolationException> {
            DTypeConstraintResolutionPass().apply(g)
        }
        val msg = ex.message ?: ""
        assertTrue(msg.contains("OneOf"), msg)
        assertTrue(msg.contains("Float32"), msg)
    }

    @Test
    fun resolved_nodes_keep_their_edges() {
        // Marking a node used to replace it in place, which dropped every edge connected to it.
        val g = DefaultComputeGraph()
        val a = g.addNode(node("a", opName = "input"))
        val b = g.addNode(node("b", policy = DTypePolicy.Prefer(FP32)))
        val c = g.addNode(node("c", opName = "relu"))
        g.addEdge(GraphEdge("ab", a, b, tensorSpec = a.outputs.single()))
        g.addEdge(GraphEdge("bc", b, c, tensorSpec = b.outputs.single()))

        val result = DTypeConstraintResolutionPass().apply(g)

        assertTrue(result.changed)
        assertEquals(listOf("a" to "b", "b" to "c"), result.graph.edges.map { it.source.id to it.destination.id })
        assertEquals(listOf("a", "b", "c"), result.graph.getTopologicalOrder().map { it.id })
        val marked = result.graph.nodes.single { it.id == "b" }
        assertEquals(true, marked.metadata[DTypeConstraintResolutionPass.RESOLVED_KEY])
        assertTrue(result.graph.edges.all { it.source in result.graph.nodes && it.destination in result.graph.nodes })
    }

    @Test
    fun a_second_run_changes_nothing() {
        val g = DefaultComputeGraph()
        g.addNode(node("n0", policy = DTypePolicy.Require(FP32)))

        val first = DTypeConstraintResolutionPass().apply(g)
        val second = DTypeConstraintResolutionPass().apply(first.graph)

        assertTrue(first.changed)
        assertFalse(second.changed, "an already resolved graph must let a pipeline converge")
    }

    @Test
    fun boundary_policy_applies_to_float_inputs_only() {
        // An op recorded inside a precision boundary: its float input is checked against the
        // declared dtype, an index input is not.
        fun boundaryNode(floatDtype: String) = GraphNode(
            id = "g",
            operation = GenericOperation("gather"),
            inputs = listOf(
                TensorSpec(name = "table", shape = listOf(8, 4), dtype = floatDtype),
                TensorSpec(name = "ids", shape = listOf(2), dtype = "Int32"),
            ),
            outputs = listOf(TensorSpec(name = "rows", shape = listOf(2, 4), dtype = floatDtype)),
            metadata = mapOf(
                DTypeConstraintResolutionPass.POLICY_KEY to DTypePolicy.Prefer(BF16),
                DTypeConstraintResolutionPass.BOUNDARY_KEY to "encoder",
            ),
        )

        val kept = DefaultComputeGraph().also { it.addNode(boundaryNode("BF16")) }
        assertEquals(emptyList(), DTypeConstraintResolutionPass().apply(kept).diagnostics)

        val retyped = DefaultComputeGraph().also { it.addNode(boundaryNode("FP32")) }
        val diagnostics = DTypeConstraintResolutionPass().apply(retyped).diagnostics
        assertEquals(1, diagnostics.size, diagnostics.toString())
        assertTrue("in precision boundary 'encoder'" in diagnostics.single(), diagnostics.single())
        assertTrue("input 0" in diagnostics.single() && "FP32" in diagnostics.single(), diagnostics.single())
    }
}
