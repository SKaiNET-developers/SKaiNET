package sk.ainet.lang.graph

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP32

class ResolvedComputeGraphTest {

    private fun makeGraph(
        edgeDtype: String = "Float32",
        nodeMarkedResolved: Boolean = true,
    ): ComputeGraph {
        val g = DefaultComputeGraph()
        val meta = if (nodeMarkedResolved) mapOf<String, Any>("dtype_resolved" to true) else emptyMap()
        val a = g.addNode(GraphNode(
            id = "a", operation = GenericOperation("input"),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a-out", listOf(4), edgeDtype)),
            metadata = meta,
        ))
        val b = g.addNode(GraphNode(
            id = "b", operation = GenericOperation("noop"),
            inputs = listOf(TensorSpec("a-out", listOf(4), edgeDtype)),
            outputs = listOf(TensorSpec("b-out", listOf(4), edgeDtype)),
            metadata = meta,
        ))
        g.addEdge(GraphEdge("e1", a, b, tensorSpec = TensorSpec("e1", listOf(4), edgeDtype)))
        return g
    }

    @Test
    fun resolvedDtype_decodes_canonical_name() {
        val g = ResolvedComputeGraph(makeGraph(edgeDtype = "Float32"))
        assertEquals(FP32, g.resolvedDtype("e1"))
    }

    @Test
    fun resolvedDtype_decodes_short_alias() {
        val g = ResolvedComputeGraph(makeGraph(edgeDtype = "BF16"))
        assertEquals(BF16, g.resolvedDtype("e1"))
    }

    @Test
    fun resolvedDtype_returns_null_for_unknown_edge() {
        val g = ResolvedComputeGraph(makeGraph())
        assertNull(g.resolvedDtype("does-not-exist"))
    }

    @Test
    fun layout_and_backend_are_placeholders() {
        val g = ResolvedComputeGraph(makeGraph())
        assertNull(g.resolvedLayout("e1"), "layout placeholder must return null today")
        assertNull(g.backendAssignment("a"), "backend placeholder must return null today")
    }

    @Test
    fun validate_passes_for_well_formed_graph() {
        val g = ResolvedComputeGraph(makeGraph())
        val result = g.validate()
        assertTrue(result.valid)
        assertEquals(emptyList(), result.errors)
    }

    @Test
    fun validate_fails_for_missing_resolved_marker() {
        val g = ResolvedComputeGraph(makeGraph(nodeMarkedResolved = false))
        val result = g.validate()
        assertFalse(result.valid)
        assertTrue(
            result.errors.any { it.contains("dtype_resolved") },
            "missing-marker error must mention the marker key: ${result.errors}",
        )
    }

    @Test
    fun validate_fails_for_unparseable_dtype() {
        val g = ResolvedComputeGraph(makeGraph(edgeDtype = "imaginary"))
        val result = g.validate()
        assertFalse(result.valid)
        assertTrue(
            result.errors.any { it.contains("imaginary") },
            "errors must surface the unparseable dtype string: ${result.errors}",
        )
    }

    @Test
    fun requireValid_throws_on_invalid_graph() {
        val g = ResolvedComputeGraph(makeGraph(nodeMarkedResolved = false))
        assertFailsWith<IllegalArgumentException> { g.validate().requireValid() }
    }
}
