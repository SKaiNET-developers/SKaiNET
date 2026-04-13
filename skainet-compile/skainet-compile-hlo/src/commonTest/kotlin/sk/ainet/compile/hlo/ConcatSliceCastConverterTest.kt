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
 * Covers #489: concat / slice (in ShapeOperationsConverter) and
 * cast (in MathOperationsConverter). All three are generic
 * structural / type primitives — concat glues tensors along an
 * axis, slice extracts a static window, cast reinterprets the
 * element type. None of them are LLM-specific; they're the
 * standard companions to reshape / flatten / squeeze that were
 * already covered.
 */
class ConcatSliceCastConverterTest {

    // ----- concat ------------------------------------------------------------

    @Test
    fun concat_and_aliases_are_supported() {
        for (opName in listOf("concat", "concatenate", "cat", "stack")) {
            val module = buildConcatModule(opName)
            assertFalse(
                module.content.contains("Unsupported operation"),
                "$opName must be claimed by a converter"
            )
            assertTrue(
                module.content.contains("stablehlo.concatenate"),
                "$opName must lower to stablehlo.concatenate"
            )
        }
    }

    @Test
    fun concat_emits_dim_attribute_matching_axis_parameter() {
        val module = buildConcatModule("concat", axis = 1)
        assertTrue(
            module.content.contains("dim = 1"),
            "concat must emit `dim = <axis>` on stablehlo.concatenate"
        )
    }

    private fun buildConcatModule(opName: String, axis: Int = 0): StableHloModule {
        val graph = DefaultComputeGraph()
        val shape = listOf(2, 3)

        val a = GraphNode(
            id = "a",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", shape, "FP32"))
        )
        val b = GraphNode(
            id = "b",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", shape, "FP32"))
        )
        val outShape = shape.mapIndexed { i, d -> if (i == axis) d * 2 else d }
        val concat = GraphNode(
            id = "cat1",
            operation = concatOp(opName, axis),
            inputs = listOf(
                TensorSpec("a", shape, "FP32"),
                TensorSpec("b", shape, "FP32")
            ),
            outputs = listOf(TensorSpec("y", outShape, "FP32"))
        )

        graph.addNode(a)
        graph.addNode(b)
        graph.addNode(concat)
        graph.addEdge(GraphEdge("e1", a, concat, 0, 0, a.outputs[0]))
        graph.addEdge(GraphEdge("e2", b, concat, 0, 1, b.outputs[0]))

        return StableHloConverterFactory.createExtended().convert(graph, "test_$opName")
    }

    // ----- slice -------------------------------------------------------------

    @Test
    fun slice_is_supported_and_emits_stablehlo_slice() {
        val module = buildSliceModule()
        assertFalse(
            module.content.contains("Unsupported operation slice"),
            "slice must be claimed by a converter"
        )
        assertTrue(
            module.content.contains("stablehlo.slice"),
            "slice must lower to stablehlo.slice"
        )
    }

    @Test
    fun slice_carries_start_limit_stride_attributes() {
        val module = buildSliceModule()
        println("[DEBUG_LOG] slice export:\n${module.content}")
        assertTrue(
            module.content.contains("start_indices"),
            "slice must emit start_indices"
        )
        assertTrue(
            module.content.contains("limit_indices"),
            "slice must emit limit_indices"
        )
        assertTrue(
            module.content.contains("strides"),
            "slice must emit strides"
        )
    }

    private fun buildSliceModule(): StableHloModule {
        val graph = DefaultComputeGraph()
        val shape = listOf(8, 16)

        val x = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", shape, "FP32"))
        )
        val slice = GraphNode(
            id = "slice1",
            operation = sliceOp(
                starts = listOf(0, 0),
                limits = listOf(4, 8),
                strides = listOf(1, 1)
            ),
            inputs = listOf(TensorSpec("x", shape, "FP32")),
            outputs = listOf(TensorSpec("y", listOf(4, 8), "FP32"))
        )
        graph.addNode(x)
        graph.addNode(slice)
        graph.addEdge(GraphEdge("e1", x, slice, 0, 0, x.outputs[0]))

        return StableHloConverterFactory.createExtended().convert(graph, "test_slice")
    }

    // ----- cast --------------------------------------------------------------

    @Test
    fun cast_and_aliases_are_supported() {
        for (opName in listOf("cast", "convert", "to")) {
            val module = buildCastModule(opName, toDtype = "FP16")
            assertFalse(
                module.content.contains("Unsupported operation"),
                "$opName must be claimed by a converter"
            )
            assertTrue(
                module.content.contains("stablehlo.convert"),
                "$opName must lower to stablehlo.convert"
            )
        }
    }

    @Test
    fun cast_emits_dtype_transition_in_type_signature() {
        val module = buildCastModule("cast", toDtype = "FP16")
        // The emitted op must carry a type signature that shows the
        // source and destination element types. Exact formatting
        // comes from TypeMapper; we check for the target dtype's
        // MLIR-style name appearing on the RHS of a `->`.
        assertTrue(
            module.content.contains("->"),
            "cast must emit a type-transition arrow in its signature"
        )
        assertTrue(
            module.content.contains("f16"),
            "cast to FP16 must mention the target element type f16"
        )
    }

    private fun buildCastModule(opName: String, toDtype: String): StableHloModule {
        val graph = DefaultComputeGraph()
        val shape = listOf(2, 3)

        val x = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", shape, "FP32"))
        )
        val cast = GraphNode(
            id = "cast1",
            operation = castOp(opName, toDtype),
            inputs = listOf(TensorSpec("x", shape, "FP32")),
            outputs = listOf(TensorSpec("y", shape, toDtype))
        )
        graph.addNode(x)
        graph.addNode(cast)
        graph.addEdge(GraphEdge("e1", x, cast, 0, 0, x.outputs[0]))

        return StableHloConverterFactory.createExtended().convert(graph, "test_$opName")
    }

    // ----- fixtures ----------------------------------------------------------

    private fun markerInputOp(): Operation = object : Operation {
        override val name: String = "input"
        override val type: String = "input"
        override val parameters: Map<String, Any> = emptyMap()
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
    }

    private fun concatOp(name: String, axis: Int): Operation = object : Operation {
        override val name: String = name
        override val type: String = "shape"
        override val parameters: Map<String, Any> = mapOf("axis" to axis)
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type, "parameters" to parameters)
    }

    private fun sliceOp(starts: List<Int>, limits: List<Int>, strides: List<Int>): Operation = object : Operation {
        override val name: String = "slice"
        override val type: String = "shape"
        override val parameters: Map<String, Any> = mapOf(
            "start_indices" to starts,
            "limit_indices" to limits,
            "strides" to strides
        )
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type, "parameters" to parameters)
    }

    private fun castOp(name: String, toDtype: String): Operation = object : Operation {
        override val name: String = name
        override val type: String = "math"
        override val parameters: Map<String, Any> = mapOf("to" to toDtype)
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type, "parameters" to parameters)
    }
}
