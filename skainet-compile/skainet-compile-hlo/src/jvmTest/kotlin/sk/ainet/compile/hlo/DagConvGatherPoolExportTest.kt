package sk.ainet.compile.hlo

import sk.ainet.lang.dag.avgPool2d
import sk.ainet.lang.dag.conv1d
import sk.ainet.lang.dag.dag
import sk.ainet.lang.dag.flatten
import sk.ainet.lang.dag.gather
import sk.ainet.lang.dag.maxPool2d
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Remaining post-#674 DAG-DSL export bugs (tracked under the #674 follow-up issue).
 *
 * #674 fixed reshape/matmul/concat output-spec inference. These ops still declare a
 * result/return type that contradicts the value they produce (conv/gather: `inferDagOutputSpecs`
 * has no shape rule for them; pooling: also emits `reduce_window` in a form IREE rejects).
 * All RED on develop after #674; lock for the follow-up fix.
 */
class DagConvGatherPoolExportTest {

    private fun lower(name: String, build: sk.ainet.lang.dag.DagBuilder.() -> Unit): String =
        StableHloConverterFactory.createExtended().convert(dag(build).toComputeGraph(), name).content

    @Test
    fun conv1d_declares_inferred_output_channels_and_length() {
        // input (1,3,8), weight (4,3,3) stride 1 pad 0 -> (1,4,6). Currently declared 1x3x8.
        val mlir = lower("op_conv1d") {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 3, 8), "FP32"))
            val w = input<FP32>("w", TensorSpec("w", listOf(4, 3, 3), "FP32"))
            val b = input<FP32>("b", TensorSpec("b", listOf(4), "FP32"))
            output(conv1d(x, w, b, 1, 0, 1, 1))
        }
        assertTrue(mlir.contains("-> tensor<1x4x6xf32>"), "conv1d result must be inferred 1x4x6:\n$mlir")
        assertFalse(Regex("""return %\w+ : tensor<1x3x8xf32>""").containsMatchIn(mlir), "return must not echo the input shape:\n$mlir")
    }

    @Test
    fun gather_declares_inferred_rows() {
        // table (8,4), 3 indices -> (3,4). Currently declared 8x4.
        val mlir = lower("op_gather") {
            val t = input<FP32>("t", TensorSpec("t", listOf(8, 4), "FP32"))
            val idx = input<Int32>("idx", TensorSpec("idx", listOf(3), "INT32"))
            output(gather(t, idx, 0))
        }
        assertTrue(mlir.contains("-> tensor<3x4xf32>"), "gather result must be inferred 3x4:\n$mlir")
        assertFalse(Regex("""return %\w+ : tensor<8x4xf32>""").containsMatchIn(mlir), "return must not echo the table shape:\n$mlir")
    }

    @Test
    fun maxpool2d_declares_pooled_shape_and_iree_valid_reduce_window() {
        // input (1,3,8,8), 2x2 stride 2 -> (1,3,4,4). Currently declared 1x3x8x8.
        val mlir = lower("op_maxpool2d") {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 3, 8, 8), "FP32"))
            output(maxPool2d(x, 2 to 2, 2 to 2, 0 to 0))
        }
        assertTrue(mlir.contains("tensor<1x3x4x4xf32>"), "maxpool output must be the pooled 1x3x4x4:\n$mlir")
        // IREE's parser rejects the pretty `applies … over window` form; it needs the generic
        // region-based reduce_window. Assert we are not emitting the rejected pretty form.
        assertFalse(
            Regex("""reduce_window\([^)]*\)\s+applies""").containsMatchIn(mlir),
            "reduce_window must use the IREE-parseable generic form, not 'applies … over window':\n$mlir",
        )
    }

    @Test
    fun flatten_preserves_leading_batch_dim() {
        // (1,16,7,7) flatten dims 1..3 -> (1, 784); must NOT collapse to rank-1 (784),
        // which breaks a downstream dense matmul (mnist-cnn).
        val mlir = lower("op_flatten") {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 16, 7, 7), "FP32"))
            output(flatten(x, 1, 3))
        }
        assertTrue(mlir.contains("tensor<1x784xf32>"), "flatten must keep batch: (1,16,7,7)->(1,784):\n$mlir")
        assertFalse(Regex("""-> tensor<784xf32>""").containsMatchIn(mlir), "flatten must not collapse the batch dim:\n$mlir")
    }

    @Test
    fun avgpool2d_declares_pooled_shape() {
        val mlir = lower("op_avgpool2d") {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 3, 8, 8), "FP32"))
            output(avgPool2d(x, 2 to 2, 2 to 2, 0 to 0, false))
        }
        assertTrue(mlir.contains("tensor<1x3x4x4xf32>"), "avgpool output must be the pooled 1x3x4x4:\n$mlir")
    }
}
