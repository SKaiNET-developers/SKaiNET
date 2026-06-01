package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Covers the converters added for the transformer (gemma3) export path:
 *  - permute -> stablehlo.transpose (arbitrary axes)
 *  - narrow(dim,start,length) -> single-axis stablehlo.slice
 */
class NarrowPermuteConverterTest {

    private fun opNode(
        id: String,
        opName: String,
        opType: String,
        params: Map<String, Any>,
        input: TensorSpec,
        output: TensorSpec,
    ): GraphNode = GraphNode(
        id = id,
        operation = object : Operation {
            override val name = opName
            override val type = opType
            override val parameters = params
            override fun <T : DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> =
                throw UnsupportedOperationException("test op")
            override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult =
                sk.ainet.lang.tensor.ops.ValidationResult.Valid
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = listOf(output)
            override fun clone(newParameters: Map<String, Any>): Operation = this
            override fun serialize(): Map<String, Any> = params
        },
        inputs = listOf(input),
        outputs = listOf(output),
    )

    @Test
    fun permuteLowersToTranspose() {
        val g = DefaultComputeGraph()
        val a = GraphNode("a", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("a", listOf(2, 3, 4), "FP32")))
        val p = opNode(
            "p", "permute", "linalg", mapOf("axes" to listOf(0, 2, 1)),
            TensorSpec("a", listOf(2, 3, 4), "FP32"), TensorSpec("b", listOf(2, 4, 3), "FP32"),
        )
        g.addNode(a); g.addNode(p)
        g.addEdge(GraphEdge("e1", a, p, 0, 0, a.outputs[0]))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "permute_test").content
        assertTrue(mlir.contains("stablehlo.transpose"), "expected transpose in:\n$mlir")
        assertTrue(mlir.contains("dims = [0, 2, 1]"), "expected axes [0,2,1] in:\n$mlir")
    }

    @Test
    fun narrowLowersToSlice() {
        val g = DefaultComputeGraph()
        val a = GraphNode("a", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("a", listOf(2, 8), "FP32")))
        val n = opNode(
            "n", "narrow", "shape", mapOf("dim" to 1, "start" to 2, "length" to 4),
            TensorSpec("a", listOf(2, 8), "FP32"), TensorSpec("b", listOf(2, 4), "FP32"),
        )
        g.addNode(a); g.addNode(n)
        g.addEdge(GraphEdge("e1", a, n, 0, 0, a.outputs[0]))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "narrow_test").content
        assertTrue(mlir.contains("stablehlo.slice"), "expected slice in:\n$mlir")
        assertTrue(mlir.contains("start_indices = [0, 2]"), "expected start [0,2] in:\n$mlir")
        assertTrue(mlir.contains("limit_indices = [2, 6]"), "expected limit [2,6] in:\n$mlir")
    }
}
