package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Covers the #1247 strictness contract: conversion failures must abort with
 * a diagnosable exception instead of degrading into MLIR comments plus an
 * empty `return` that exits 0. [ConversionErrorPolicy.LENIENT] preserves the
 * historical comment-and-continue behavior for callers that inspect
 * partially-converted modules.
 */
class ConversionStrictnessTest {

    // --- resolveOperands ---------------------------------------------------

    @Test
    fun strict_resolveOperands_throws_named_exception_for_unresolved_producer() {
        val (graph, _, consumer) = twoNodeGraph()
        val context = ConversionContext(TypeMapper(), graph)
        // The producer was never converted: no SSA name registered for it.
        val e = assertFailsWith<MissingOperandException> {
            context.resolveOperands(consumer)
        }
        assertEquals("consumer", e.nodeId)
        assertEquals("relu", e.opName)
        assertEquals(0, e.inputPort)
        assertEquals("producer", e.sourceNodeId)
    }

    @Test
    fun lenient_resolveOperands_reproduces_silent_drop_and_positional_shift() {
        // Two producers into ports 0 and 1; only port 1's producer resolved.
        val graph = DefaultComputeGraph()
        val p0 = inputNode("p0")
        val p1 = inputNode("p1")
        val consumer = opNode("consumer", "gather", inputs = 2)
        graph.addNode(p0)
        graph.addNode(p1)
        graph.addNode(consumer)
        graph.addEdge(GraphEdge("e0", p0, consumer, 0, 0, p0.outputs[0]))
        graph.addEdge(GraphEdge("e1", p1, consumer, 0, 1, p1.outputs[0]))

        val context = ConversionContext(
            TypeMapper(), graph,
            errorPolicy = ConversionErrorPolicy.LENIENT
        )
        context.setValueName("p1", "%arg1")

        // Historical defect, now an explicit contract of LENIENT: the
        // unresolved port-0 operand is dropped and port 1 slides into slot 0.
        assertEquals(listOf("%arg1"), context.resolveOperands(consumer))
    }

    // --- converter-level strictness ----------------------------------------

    @Test
    fun strict_convert_throws_on_unknown_op_with_node_and_registry_diagnostics() {
        val graph = singleOpGraph(opName = "definitelyNotAnOp")
        val converter = StableHloConverterFactory.createExtended()
        val e = assertFailsWith<HloConversionException> {
            converter.convert(graph, "strict_unknown")
        }
        val message = e.message ?: ""
        assertTrue("definitelyNotAnOp" in message, "exception must name the failing op: $message")
        assertTrue("op1" in message, "exception must name the failing node: $message")
        assertTrue("Registry known names" in message, "exception must carry the registry key set: $message")
    }

    @Test
    fun lenient_convert_keeps_comment_fallback_and_exits_normally() {
        val graph = singleOpGraph(opName = "definitelyNotAnOp")
        val converter = StableHloConverterFactory.createExtended(
            errorPolicy = ConversionErrorPolicy.LENIENT
        )
        val module = converter.convert(graph, "lenient_unknown")
        assertTrue(module.content.contains("Error processing node op1"))
        assertTrue(module.content.contains("Known names:"))
    }

    @Test
    fun strict_convert_throws_on_conversion_failure_result() {
        // A gather wired with only one operand: resolveOperands succeeds (the
        // one producer is an input) but the converter returns Failure — which
        // must now throw instead of emitting a comment.
        val graph = DefaultComputeGraph()
        val weight = inputNode("W", shape = listOf(8, 4))
        val gather = opNode("embed1", "gather", inputs = 1)
        graph.addNode(weight)
        graph.addNode(gather)
        graph.addEdge(GraphEdge("e1", weight, gather, 0, 0, weight.outputs[0]))

        val converter = StableHloConverterFactory.createExtended()
        val e = assertFailsWith<HloConversionException> {
            converter.convert(graph, "strict_arity")
        }
        val message = e.message ?: ""
        assertTrue("embed1" in message, "exception must name the failing node: $message")
        assertTrue("2 operands" in message, "exception must carry the converter's error: $message")
    }

    // --- createBasic registry parity (#1247: gemma3n harness uses createBasic) ---

    @Test
    fun createBasic_routes_neural_net_ops() {
        // Previously createBasic lacked NeuralNetOperationsConverter, so a
        // traced model with norms could not lower. "No converter found" (a
        // registry miss) must not appear for rmsNorm; a Failure from the
        // converter itself would surface differently and is acceptable here.
        val graph = singleOpGraph(opName = "rmsNorm")
        val converter = StableHloConverterFactory.createBasic(
            errorPolicy = ConversionErrorPolicy.LENIENT
        )
        val module = converter.convert(graph, "basic_rmsnorm")
        assertFalse(
            module.content.contains("No converter found for operation: rmsNorm"),
            "createBasic must register NeuralNetOperationsConverter:\n${module.content}"
        )
    }

    // --- fixtures ----------------------------------------------------------

    private fun twoNodeGraph(): Triple<DefaultComputeGraph, GraphNode, GraphNode> {
        val graph = DefaultComputeGraph()
        val producer = opNode("producer", "mysteryOp", inputs = 0)
        val consumer = opNode("consumer", "relu", inputs = 1)
        graph.addNode(producer)
        graph.addNode(consumer)
        graph.addEdge(GraphEdge("e1", producer, consumer, 0, 0, producer.outputs[0]))
        return Triple(graph, producer, consumer)
    }

    private fun singleOpGraph(opName: String): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val input = inputNode("in1")
        val op = opNode("op1", opName, inputs = 1)
        graph.addNode(input)
        graph.addNode(op)
        graph.addEdge(GraphEdge("e1", input, op, 0, 0, input.outputs[0]))
        return graph
    }

    private fun inputNode(id: String, shape: List<Int> = listOf(2, 2)): GraphNode = GraphNode(
        id = id,
        operation = fixtureOp("input", "input"),
        inputs = emptyList(),
        outputs = listOf(TensorSpec(id, shape, "FP32"))
    )

    private fun opNode(id: String, name: String, inputs: Int): GraphNode = GraphNode(
        id = id,
        operation = fixtureOp(name, "compute"),
        inputs = List(inputs) { TensorSpec("in$it", listOf(2, 2), "FP32") },
        outputs = listOf(TensorSpec("$id-out", listOf(2, 2), "FP32"))
    )

    private fun fixtureOp(opName: String, opType: String): Operation = object : Operation {
        override val name: String = opName
        override val type: String = opType
        override val parameters: Map<String, Any> = emptyMap()
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to opName, "type" to opType)
    }
}
