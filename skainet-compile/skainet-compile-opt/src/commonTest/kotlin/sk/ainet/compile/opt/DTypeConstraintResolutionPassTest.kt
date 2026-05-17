package sk.ainet.compile.opt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.opt.passes.DTypeConstraintResolutionPass
import sk.ainet.compile.opt.passes.DtypeConstraintViolationException
import sk.ainet.lang.graph.DefaultComputeGraph
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
}
