package sk.ainet.compile.opt

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.opt.passes.PowSpecializationPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.tensor.ops.PowOperation
import sk.ainet.lang.tensor.ops.TensorSpec

class PowSpecializationPassTest {

    private fun spec(name: String = "t") = TensorSpec(name = name, shape = listOf(4), dtype = "Float32")

    @Test
    fun rewrites_pow_x_2_to_multiply_x_x() {
        val g = DefaultComputeGraph()
        val input = g.addNode(
            GraphNode(id = "x", operation = GenericOperation("input"), inputs = emptyList(), outputs = listOf(spec("x"))),
        )
        val pow = g.addNode(
            GraphNode(
                id = "pow1",
                operation = PowOperation<sk.ainet.lang.types.DType, Any>(
                    parameters = mapOf("scalar_exponent" to 2),
                ),
                inputs = listOf(spec("x")),
                outputs = listOf(spec("pow_out")),
            ),
        )
        g.addEdge(GraphEdge("e0", input, pow, tensorSpec = spec()))

        val result = PowSpecializationPass().apply(g)
        assertTrue(result.changed, "pass must report changed=true")

        // The original pow node is gone; in its place there's a multiply
        // with both inputs routed to x.
        val mul = result.graph.nodes.firstOrNull { it.id == "pow1" }
        assertTrue(mul != null && mul.operation is MultiplyOperation<*, *>, "node 'pow1' must now be a multiply")
        val mulIncoming = result.graph.edges.filter { it.destination.id == "pow1" }
        assertEquals(2, mulIncoming.size, "multiply must have two incoming edges")
        assertTrue(mulIncoming.all { it.source.id == "x" }, "both multiply inputs route to x")
    }

    @Test
    fun leaves_pow_x_3_untouched_in_first_cut() {
        // Tier A only specialises n=2; n=3 + higher are follow-ups.
        val g = DefaultComputeGraph()
        val input = g.addNode(GraphNode("x", GenericOperation("input"), emptyList(), listOf(spec("x"))))
        g.addNode(
            GraphNode(
                "pow1",
                PowOperation<sk.ainet.lang.types.DType, Any>(parameters = mapOf("scalar_exponent" to 3)),
                listOf(spec("x")),
                listOf(spec("pow_out")),
            ),
        )
        g.addEdge(GraphEdge("e0", input, g.nodes.first { it.id == "pow1" }, tensorSpec = spec()))

        val result = PowSpecializationPass().apply(g)
        assertFalse(result.changed, "pass must skip n != 2 in first cut")
        val pow = result.graph.nodes.first { it.id == "pow1" }
        assertTrue(pow.operation is PowOperation<*, *>, "node must remain a PowOperation")
    }

    @Test
    fun leaves_pow_binary_form_untouched() {
        // PowOperation with two inputs (tensor exponent) is not a scalar-pow
        // case — pass must ignore it.
        val g = DefaultComputeGraph()
        val a = g.addNode(GraphNode("a", GenericOperation("input"), emptyList(), listOf(spec("a"))))
        val b = g.addNode(GraphNode("b", GenericOperation("input"), emptyList(), listOf(spec("b"))))
        val pow = g.addNode(
            GraphNode(
                "pow1",
                PowOperation<sk.ainet.lang.types.DType, Any>(),
                listOf(spec("a"), spec("b")),
                listOf(spec("pow_out")),
            ),
        )
        g.addEdge(GraphEdge("e0", a, pow, destinationInputIndex = 0, tensorSpec = spec()))
        g.addEdge(GraphEdge("e1", b, pow, destinationInputIndex = 1, tensorSpec = spec()))

        val result = PowSpecializationPass().apply(g)
        assertFalse(result.changed, "pass must skip binary pow")
    }

    @Test
    fun leaves_graphs_without_pow_untouched() {
        val g = DefaultComputeGraph()
        g.addNode(GraphNode("x", GenericOperation("input"), emptyList(), listOf(spec("x"))))
        g.addNode(GraphNode("relu", GenericOperation("relu"), listOf(spec("x")), listOf(spec("relu_out"))))
        val result = PowSpecializationPass().apply(g)
        assertFalse(result.changed)
    }
}
