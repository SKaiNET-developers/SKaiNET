package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.ResolvedComputeGraph
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Round-trip and byte-equivalence tests for the
 * `toStableHlo(ResolvedComputeGraph)` overload added in W9 of #615.
 *
 * Key property: when the underlying graph passes resolved-graph
 * validation, the two HLO entry points must produce byte-identical
 * output. The wrapper is the *contract*, not a separate emit path.
 */
class ResolvedComputeGraphHloTest {

    private fun buildSimpleGraph(): DefaultComputeGraph {
        val g = DefaultComputeGraph()
        val resolvedMeta = mapOf<String, Any>("dtype_resolved" to true)
        val input = g.addNode(
            GraphNode(
                id = "input",
                operation = GenericOperation("input"),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("input-out", listOf(2, 3), "FP32")),
                metadata = resolvedMeta,
            ),
        )
        val relu = g.addNode(
            GraphNode(
                id = "relu",
                operation = GenericOperation("relu"),
                inputs = listOf(TensorSpec("input-out", listOf(2, 3), "FP32")),
                outputs = listOf(TensorSpec("relu-out", listOf(2, 3), "FP32")),
                metadata = resolvedMeta,
            ),
        )
        g.addEdge(GraphEdge("e1", input, relu, tensorSpec = TensorSpec("e1", listOf(2, 3), "FP32")))
        return g
    }

    @Test
    fun resolved_overload_produces_same_module_as_plain_overload() {
        val g = buildSimpleGraph()
        val viaPlain = toStableHlo(g)
        val viaResolved = toStableHlo(ResolvedComputeGraph(g))
        // functionName + content are byte-identical.
        assertEquals(viaPlain.functionName, viaResolved.functionName)
        assertEquals(viaPlain.content, viaResolved.content)
    }

    @Test
    fun resolved_overload_validates_by_default() {
        // Build a graph that's missing the dtype_resolved marker.
        val g = DefaultComputeGraph()
        val node = g.addNode(
            GraphNode(
                id = "input",
                operation = GenericOperation("input"),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("input-out", listOf(2), "FP32")),
                metadata = emptyMap(),
            ),
        )
        // Default validate = true must reject this.
        assertFailsWith<IllegalArgumentException> {
            toStableHlo(ResolvedComputeGraph(g))
        }
    }

    @Test
    fun resolved_overload_skips_validation_when_opted_out() {
        // Same unmarked graph, but validate = false.
        val g = DefaultComputeGraph()
        g.addNode(
            GraphNode(
                id = "input",
                operation = GenericOperation("input"),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("input-out", listOf(2), "FP32")),
                metadata = emptyMap(),
            ),
        )
        // No throw expected.
        val module = toStableHlo(ResolvedComputeGraph(g), validate = false)
        assertEquals("main", module.functionName)
    }
}
