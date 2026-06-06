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
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Regression tests for the ShapeOperationsConverter shape bugs:
 *  - #667: multi-input `concatenate` must SUM the operands' extents on the axis.
 *  - #666: `reshape` whose target shape lives only in a parameter must still lower.
 */
class ReshapeConcatShapeFixTest {

    private fun op(name: String, type: String, params: Map<String, Any> = emptyMap()): Operation =
        object : Operation {
            override val name: String = name
            override val type: String = type
            override val parameters: Map<String, Any> = params
            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
                throw UnsupportedOperationException("marker")
            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
            override fun clone(newParameters: Map<String, Any>): Operation = this
            override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
        }

    @Test
    fun multiInputConcat_sums_the_concatenated_axis() {
        val g = DefaultComputeGraph()
        val a = GraphNode("a", op("input", "input"), emptyList(), listOf(TensorSpec("a", listOf(1, 1, 8, 8), "FP32")))
        val b = GraphNode("b", op("input", "input"), emptyList(), listOf(TensorSpec("b", listOf(1, 4, 8, 8), "FP32")))
        val c = GraphNode("c", op("input", "input"), emptyList(), listOf(TensorSpec("c", listOf(1, 1, 8, 8), "FP32")))
        g.addNode(a); g.addNode(b); g.addNode(c)

        val cat = GraphNode(
            id = "cat",
            operation = op("concatenate", "shape", mapOf("axis" to 1)),
            inputs = listOf(TensorSpec("a", listOf(1, 1, 8, 8), "FP32"), TensorSpec("b", listOf(1, 4, 8, 8), "FP32"), TensorSpec("c", listOf(1, 1, 8, 8), "FP32")),
            // Deliberately WRONG declared output shape (operand-0 extent on the axis):
            outputs = listOf(TensorSpec("y", listOf(1, 1, 8, 8), "FP32")),
        )
        g.addNode(cat)
        g.addEdge(GraphEdge("e0", a, cat, 0, 0, a.outputs[0]))
        g.addEdge(GraphEdge("e1", b, cat, 0, 1, b.outputs[0]))
        g.addEdge(GraphEdge("e2", c, cat, 0, 2, c.outputs[0]))

        val mlir = StableHloConverterFactory.createExtended().convert(g, "concat3").content
        assertTrue(mlir.contains("stablehlo.concatenate"), "expected a concatenate op:\n$mlir")
        assertTrue(mlir.contains("-> tensor<1x6x8x8xf32>"), "concat result must sum the axis to 6:\n$mlir")
        assertFalse(mlir.contains("-> tensor<1x1x8x8xf32>"), "must not echo operand-0's axis extent:\n$mlir")
    }

    @Test
    fun reshape_with_shape_only_in_parameter_still_lowers() {
        val g = DefaultComputeGraph()
        val x = GraphNode("x", op("input", "input"), emptyList(), listOf(TensorSpec("x", listOf(1, 12), "FP32")))
        g.addNode(x)
        val r = GraphNode(
            id = "r",
            operation = op("reshape", "shape", mapOf("outputShape" to listOf(1, 3, 4))),
            inputs = listOf(TensorSpec("x", listOf(1, 12), "FP32")),
            outputs = emptyList(), // no declared output spec — target lives only in the param
        )
        g.addNode(r)
        g.addEdge(GraphEdge("e0", x, r, 0, 0, x.outputs[0]))

        val mlir = StableHloConverterFactory.createExtended().convert(g, "reshape1").content
        assertTrue(mlir.contains("stablehlo.reshape"), "reshape must lower (not an empty module):\n$mlir")
        assertTrue(mlir.contains("tensor<1x3x4xf32>"), "reshape must carry the target shape:\n$mlir")
    }
}
