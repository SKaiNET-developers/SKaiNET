package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import java.io.File
import kotlin.test.Test

/** Dumps the emitted dynamic-key SDPA StableHLO to `DYN_MLIR_OUT` (if set) so it can be fed to `iree-compile`
 *  to confirm the dynamic-shape-safe emission actually compiles. No-op assertion when the env var is unset. */
class DynamicShapeHloDumpTest {
    private fun op(opName: String, params: Map<String, Any> = emptyMap()): Operation = object : Operation {
        override val name = opName
        override val type = "test"
        override val parameters = params
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>) = throw UnsupportedOperationException()
        override fun validateInputs(inputs: List<TensorSpec>) = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>) = inputs
        override fun clone(newParameters: Map<String, Any>) = op(opName, newParameters)
        override fun serialize() = mapOf("name" to name)
    }

    @Test
    fun dumpDynamicSdpa() {
        val out = System.getenv("DYN_MLIR_OUT") ?: return
        val q = TensorSpec("q", listOf(1, 8, 1, 40), "FP32")
        val k = TensorSpec("k", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")
        val v = TensorSpec("v", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")
        val o = TensorSpec("out", listOf(1, 8, 1, 40), "FP32")
        val g = DefaultComputeGraph()
        val nq = GraphNode("q", InputOperation<DType, Any>(), emptyList(), listOf(q))
        val nk = GraphNode("k", InputOperation<DType, Any>(), emptyList(), listOf(k))
        val nv = GraphNode("v", InputOperation<DType, Any>(), emptyList(), listOf(v))
        val sdpa = GraphNode("sdpa", op("scaledDotProductAttention", mapOf("causal" to false)),
            listOf(q, k, v), listOf(o))
        listOf(nq, nk, nv, sdpa).forEach { g.addNode(it) }
        g.addEdge(GraphEdge("e0", nq, sdpa, 0, 0, q))
        g.addEdge(GraphEdge("e1", nk, sdpa, 0, 1, k))
        g.addEdge(GraphEdge("e2", nv, sdpa, 0, 2, v))
        File(out).writeText(StableHloConverterFactory.createBasic().convert(g, "dyn_sdpa").content)
        println("WROTE_MLIR $out")
    }
}
