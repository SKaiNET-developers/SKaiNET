package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import java.io.File
import kotlin.test.Test

/**
 * Dumps a small scaledDotProductAttention graph to StableHLO for numerical
 * validation against a NumPy reference (see docker iree-run-module + numpy).
 * Shapes [B=1, H=1, S=2, D=4]; scale = 1/sqrt(4) = 0.5.
 */
class SdpaNumericDumpTest {
    @Test
    fun dumpSdpaMlir() {
        val shape = listOf(1, 1, 2, 4)
        fun inNode(id: String) = GraphNode(id, InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec(id, shape, "FP32")))
        val q = inNode("q"); val k = inNode("k"); val v = inNode("v")
        val sdpa = GraphNode(
            id = "att",
            operation = object : Operation {
                override val name = "scaledDotProductAttention"
                override val type = "trace"
                override val parameters = mapOf<String, Any>("scale" to 0.0f, "causal" to true)
                override fun <T : DType, V2> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V2>>) =
                    throw UnsupportedOperationException("test op")
                override fun validateInputs(inputs: List<TensorSpec>) = sk.ainet.lang.tensor.ops.ValidationResult.Valid
                override fun inferOutputs(inputs: List<TensorSpec>) = listOf(TensorSpec("o", shape, "FP32"))
                override fun clone(newParameters: Map<String, Any>): Operation = this
                override fun serialize() = parameters
            },
            inputs = listOf(TensorSpec("q", shape, "FP32"), TensorSpec("k", shape, "FP32"), TensorSpec("v", shape, "FP32")),
            outputs = listOf(TensorSpec("o", shape, "FP32")),
        )
        val g = DefaultComputeGraph()
        g.addNode(q); g.addNode(k); g.addNode(v); g.addNode(sdpa)
        g.addEdge(GraphEdge("e0", q, sdpa, 0, 0, q.outputs[0]))
        g.addEdge(GraphEdge("e1", k, sdpa, 0, 1, k.outputs[0]))
        g.addEdge(GraphEdge("e2", v, sdpa, 0, 2, v.outputs[0]))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "sdpa").content
        val out = File(
            System.getProperty("sdpaMlirOut")
                ?: File(System.getProperty("java.io.tmpdir"), "skainet-mlir/sdpa.mlir").path,
        )
        out.parentFile?.mkdirs()
        out.writeText(mlir)
        println("WROTE_SDPA ${out.absolutePath}")
    }
}
