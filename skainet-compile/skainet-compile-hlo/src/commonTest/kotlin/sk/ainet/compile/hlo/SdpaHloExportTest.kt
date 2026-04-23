package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SdpaHloExportTest {

    @Test
    fun sdpa_produces_dot_general_ops() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val q = ctx.fromFloatArray<FP32, Float>(Shape(1, 2, 4, 8), FP32::class, FloatArray(64))
        val k = ctx.fromFloatArray<FP32, Float>(Shape(1, 2, 4, 8), FP32::class, FloatArray(64))
        val v = ctx.fromFloatArray<FP32, Float>(Shape(1, 2, 4, 8), FP32::class, FloatArray(64))

        @Suppress("UNCHECKED_CAST")
        val inputIds = setOf(
            ctx.session.refOf(q as Tensor<*, *>).id,
            ctx.session.refOf(k as Tensor<*, *>).id,
            ctx.session.refOf(v as Tensor<*, *>).id
        )

        val (tape, out) = ctx.record {
            ctx.ops.scaledDotProductAttention(q, k, v)
        }

        println("Output shape: ${out.shape}")

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = inputIds
        )
        val nodes = graph.getTopologicalOrder()
        println("Graph: ${nodes.size} nodes")
        println("Ops: ${nodes.map { it.operation.name }}")

        val module = StableHloConverterFactory.createExtended().convert(graph, "sdpa_test")
        println("MLIR:\n${module.content}")

        // Should contain dot_general for Q@K.T and weights@V
        assertTrue(module.content.contains("stablehlo.dot_general"), "Should have dot_general ops")

        // Should NOT contain large zero constant tensors (from raw permutation)
        // Scalar zeros (tensor<f32>) for softmax init are fine
        assertFalse(
            module.content.contains(Regex("dense<0\\.0> : tensor<\\d+x")),
            "Should not have large zero constant tensors"
        )

        // Should contain exponential (softmax decomposition)
        assertTrue(module.content.contains("stablehlo.exponential"), "Should have softmax (exponential)")
    }
}
