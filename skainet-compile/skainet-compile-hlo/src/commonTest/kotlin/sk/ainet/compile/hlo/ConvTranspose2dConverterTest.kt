package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.NeuralNetOperationsConverter
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Covers the convTranspose2d StableHLO lowering (#1371 / #1375): a transposed
 * convolution lowers to a regular `stablehlo.convolution` with the stride moved
 * to `lhs_dilate`, the kernel spatially flipped (`stablehlo.reverse`) and read
 * with `[i, o, 0, 1]` dim numbers, and the padding rewritten per axis to
 * `low = dilation*(k-1) - padding`, `high = low + outputPadding`.
 */
class ConvTranspose2dConverterTest {

    @Test
    fun convTranspose2d_operation_is_supported() {
        val registry = StableHloOperationRegistry()
        registry.register(NeuralNetOperationsConverter())
        assertTrue(registry.isSupported("convTranspose2d"))
    }

    @Test
    fun convTranspose2d_lowers_to_lhs_dilated_convolution_with_flipped_kernel() {
        // input [1,2,3,3], weight [C_in=2, C_out=3, 3, 3], stride 2, padding 1, outputPadding 1
        // -> output [1,3,6,6]; pad low = 1*(3-1) - 1 = 1, pad high = 1 + 1 = 2.
        val graph = buildConvTranspose2dGraph(
            stride = 2 to 2, padding = 1 to 1, outputPadding = 1 to 1, dilation = 1 to 1, groups = 1,
        )
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_conv_transpose2d")

        assertFalse(
            module.content.contains("Unsupported operation") || module.content.contains("No converter found"),
            "convTranspose2d must be claimed by a converter:\n${module.content}"
        )
        assertContains(module.content, "stablehlo.reverse", false, "kernel must be spatially flipped")
        assertContains(module.content, "dims = [2, 3]", false, "flip must be over the spatial dims")
        assertContains(module.content, "stablehlo.convolution")
        assertContains(module.content, "[i, o, 0, 1]", false, "kernel dim numbers must swap in/out features")
        assertContains(module.content, "lhs_dilate = [2, 2]", false, "stride must become lhs dilation")
        assertContains(module.content, "stride = [1, 1]", false, "window stride must be 1")
        assertContains(module.content, "pad = [[1, 2], [1, 2]]", false, "padding must be rewritten with outputPadding on the high side")
    }

    @Test
    fun convTranspose2d_grouped_lowering_is_rejected() {
        val graph = buildConvTranspose2dGraph(
            stride = 1 to 1, padding = 0 to 0, outputPadding = 0 to 0, dilation = 1 to 1, groups = 2,
        )
        val converter = StableHloConverterFactory.createExtended()
        val result = runCatching { converter.convert(graph, "test_conv_transpose2d_grouped") }

        // Either the conversion throws or the module records the failure — it must not
        // silently emit a groups-ignoring convolution.
        val content = result.getOrNull()?.content
        assertTrue(
            result.isFailure || content == null || !content.contains("lhs_dilate"),
            "grouped convTranspose2d must not silently lower as groups=1:\n$content"
        )
    }

    private fun buildConvTranspose2dGraph(
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        outputPadding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int,
    ): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val inputSpec = TensorSpec("input", listOf(1, 2, 3, 3), "FP32")
        val weightSpec = TensorSpec("weight", listOf(2, 3, 3, 3), "FP32")
        val outH = (3 - 1) * stride.first - 2 * padding.first + dilation.first * 2 + outputPadding.first + 1
        val outW = (3 - 1) * stride.second - 2 * padding.second + dilation.second * 2 + outputPadding.second + 1
        val outputSpec = TensorSpec("out", listOf(1, 3 * groups, outH, outW), "FP32")

        val input = GraphNode("input", markerInputOp(), emptyList(), listOf(inputSpec))
        val weight = GraphNode("weight", markerInputOp(), emptyList(), listOf(weightSpec))
        val convT = GraphNode(
            id = "convTranspose2d",
            operation = convTranspose2dOp(stride, padding, outputPadding, dilation, groups),
            inputs = listOf(inputSpec, weightSpec),
            outputs = listOf(outputSpec)
        )
        graph.addNode(input)
        graph.addNode(weight)
        graph.addNode(convT)
        graph.addEdge(GraphEdge("e1", input, convT, 0, 0, inputSpec))
        graph.addEdge(GraphEdge("e2", weight, convT, 0, 1, weightSpec))
        return graph
    }

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

    private fun convTranspose2dOp(
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        outputPadding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int,
    ): Operation = object : Operation {
        override val name: String = "convTranspose2d"
        override val type: String = "nn"
        override val parameters: Map<String, Any> = mapOf(
            "stride" to stride,
            "padding" to padding,
            "outputPadding" to outputPadding,
            "dilation" to dilation,
            "groups" to groups,
        )
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf(
            "name" to name, "type" to type, "parameters" to parameters
        )
    }
}
