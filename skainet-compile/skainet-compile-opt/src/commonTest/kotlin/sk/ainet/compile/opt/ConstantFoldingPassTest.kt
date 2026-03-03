package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.ConstantFoldingPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ConstantFoldingPassTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(4)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun constNode(id: String, values: List<Float>) = GraphNode(
        id = id,
        operation = GenericOperation("constant", mapOf("values" to values), "constant"),
        inputs = emptyList(),
        outputs = listOf(spec())
    )

    private fun binaryNode(id: String, opName: String) = GraphNode(
        id = id,
        operation = GenericOperation(opName),
        inputs = listOf(spec(), spec()),
        outputs = listOf(spec())
    )

    @Test
    fun foldsAdd() {
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(1.0f, 2.0f, 3.0f, 4.0f)))
        val b = graph.addNode(constNode("b", listOf(10.0f, 20.0f, 30.0f, 40.0f)))
        val add = graph.addNode(binaryNode("add1", "add"))
        graph.addEdge(GraphEdge("e1", a, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", b, add, destinationInputIndex = 1, tensorSpec = spec()))

        val result = ConstantFoldingPass().apply(graph)
        assertTrue(result.changed)
        assertEquals(3, result.graph.nodes.size) // a, b, add1 (now constant)

        val foldedNode = result.graph.nodes.first { it.id == "add1" }
        assertEquals("constant", foldedNode.operation.name)

        @Suppress("UNCHECKED_CAST")
        val values = foldedNode.operation.parameters["values"] as List<Float>
        assertEquals(listOf(11.0f, 22.0f, 33.0f, 44.0f), values)
    }

    @Test
    fun foldsMultiply() {
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(2.0f, 3.0f)))
        val b = graph.addNode(constNode("b", listOf(4.0f, 5.0f)))
        val mul = graph.addNode(binaryNode("mul1", "multiply"))
        graph.addEdge(GraphEdge("e1", a, mul, tensorSpec = spec(shape = listOf(2))))
        graph.addEdge(GraphEdge("e2", b, mul, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(2))))

        val result = ConstantFoldingPass().apply(graph)
        assertTrue(result.changed)

        val foldedNode = result.graph.nodes.first { it.id == "mul1" }
        @Suppress("UNCHECKED_CAST")
        val values = foldedNode.operation.parameters["values"] as List<Float>
        assertEquals(listOf(8.0f, 15.0f), values)
    }

    @Test
    fun foldsSubtract() {
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(10.0f, 20.0f)))
        val b = graph.addNode(constNode("b", listOf(3.0f, 7.0f)))
        val sub = graph.addNode(binaryNode("sub1", "subtract"))
        graph.addEdge(GraphEdge("e1", a, sub, tensorSpec = spec(shape = listOf(2))))
        graph.addEdge(GraphEdge("e2", b, sub, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(2))))

        val result = ConstantFoldingPass().apply(graph)
        assertTrue(result.changed)

        val foldedNode = result.graph.nodes.first { it.id == "sub1" }
        @Suppress("UNCHECKED_CAST")
        val values = foldedNode.operation.parameters["values"] as List<Float>
        assertEquals(listOf(7.0f, 13.0f), values)
    }

    @Test
    fun foldsDivide() {
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(10.0f, 20.0f)))
        val b = graph.addNode(constNode("b", listOf(2.0f, 5.0f)))
        val div = graph.addNode(binaryNode("div1", "divide"))
        graph.addEdge(GraphEdge("e1", a, div, tensorSpec = spec(shape = listOf(2))))
        graph.addEdge(GraphEdge("e2", b, div, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(2))))

        val result = ConstantFoldingPass().apply(graph)
        assertTrue(result.changed)

        val foldedNode = result.graph.nodes.first { it.id == "div1" }
        @Suppress("UNCHECKED_CAST")
        val values = foldedNode.operation.parameters["values"] as List<Float>
        assertEquals(listOf(5.0f, 4.0f), values)
    }

    @Test
    fun skipsDivisionByZero() {
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(10.0f)))
        val b = graph.addNode(constNode("b", listOf(0.0f)))
        val div = graph.addNode(binaryNode("div1", "divide"))
        graph.addEdge(GraphEdge("e1", a, div, tensorSpec = spec(shape = listOf(1))))
        graph.addEdge(GraphEdge("e2", b, div, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(1))))

        val result = ConstantFoldingPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun doesNotFoldNonConstantInputs() {
        // A (not constant) + B (constant) → should not fold
        val graph = DefaultComputeGraph()
        val input = graph.addNode(
            GraphNode("input", GenericOperation("input"), emptyList(), listOf(spec()))
        )
        val b = graph.addNode(constNode("b", listOf(1.0f, 2.0f, 3.0f, 4.0f)))
        val add = graph.addNode(binaryNode("add1", "add"))
        graph.addEdge(GraphEdge("e1", input, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", b, add, destinationInputIndex = 1, tensorSpec = spec()))

        val result = ConstantFoldingPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun emptyGraphUnchanged() {
        val graph = DefaultComputeGraph()
        val result = ConstantFoldingPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun chainedFoldingInSinglePass() {
        // A(const) + B(const) → C(add) → ... C is folded.
        // C(folded const) + D(const) → E(add)
        // In a single pass (topo order), E should also be folded because
        // C's result is cached as constant.
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(1.0f)))
        val b = graph.addNode(constNode("b", listOf(2.0f)))
        val c = graph.addNode(binaryNode("c", "add"))
        val d = graph.addNode(constNode("d", listOf(10.0f)))
        val e = graph.addNode(binaryNode("e", "add"))
        graph.addEdge(GraphEdge("e1", a, c, tensorSpec = spec(shape = listOf(1))))
        graph.addEdge(GraphEdge("e2", b, c, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(1))))
        graph.addEdge(GraphEdge("e3", c, e, tensorSpec = spec(shape = listOf(1))))
        graph.addEdge(GraphEdge("e4", d, e, destinationInputIndex = 1, tensorSpec = spec(shape = listOf(1))))

        val result = ConstantFoldingPass().apply(graph)
        assertTrue(result.changed)

        val foldedE = result.graph.nodes.first { it.id == "e" }
        assertEquals("constant", foldedE.operation.name)
        @Suppress("UNCHECKED_CAST")
        val values = foldedE.operation.parameters["values"] as List<Float>
        assertEquals(listOf(13.0f), values) // (1+2) + 10
    }
}
