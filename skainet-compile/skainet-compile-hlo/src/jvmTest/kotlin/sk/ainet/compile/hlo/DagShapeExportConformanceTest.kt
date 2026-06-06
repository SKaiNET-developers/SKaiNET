package sk.ainet.compile.hlo

import sk.ainet.lang.dag.concat
import sk.ainet.lang.dag.dag
import sk.ainet.lang.dag.matmul
import sk.ainet.lang.dag.reshape
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * End-to-end regression tests that exercise the REAL `sk.ainet.lang.dag` DSL path
 * (`dag { … }.toComputeGraph()` → extended converter), exactly as the conformance
 * harness builds its op modules.
 *
 * Distinct from [ReshapeConcatShapeFixTest], which constructs synthetic `GraphNode`s
 * directly: that test passes while these fail, because the bug lives in how the DAG
 * DSL records a shape-changing op's output spec — not in the converter's shape math.
 *
 * Each asserts the emitted module is IREE-compilable shape-wise: the op lowers AND the
 * declared result/return type matches the value it actually produces.
 */
class DagShapeExportConformanceTest {

    private fun lower(name: String, build: sk.ainet.lang.dag.DagBuilder.() -> Unit): String =
        StableHloConverterFactory.createExtended().convert(dag(build).toComputeGraph(), name).content

    @Test
    fun reshape_via_dag_dsl_lowers_with_target_shape() {
        // reshape (1,4) -> (2,2). Harness: OpsModel.reshapeMlir().
        val mlir = lower("op_reshape") {
            val a = input<FP32>("a", TensorSpec("a", listOf(1, 4), "FP32"))
            output(reshape(a, Shape(intArrayOf(2, 2))))
        }
        assertFalse(
            mlir.contains("Missing shape parameter") || mlir.contains("requires a target shape"),
            "reshape dropped its target shape — empty/invalid module:\n$mlir",
        )
        assertTrue(mlir.contains("stablehlo.reshape"), "reshape must lower:\n$mlir")
        assertTrue(mlir.contains("tensor<2x2xf32>"), "reshape must carry target shape 2x2:\n$mlir")
    }

    @Test
    fun concat_via_dag_dsl_propagates_summed_axis_to_return() {
        // concat([(1,4),(1,4)], dim=1) -> (1,8). Harness: OpsModel.concatMlir().
        val mlir = lower("op_concat") {
            val a = input<FP32>("a", TensorSpec("a", listOf(1, 4), "FP32"))
            val b = input<FP32>("b", TensorSpec("b", listOf(1, 4), "FP32"))
            output(concat(listOf(a, b), 1))
        }
        assertTrue(mlir.contains("-> tensor<1x8xf32>"), "concat op must type the axis-sum 1x8:\n$mlir")
        // The function return must agree with the value it returns (else iree-compile rejects it).
        assertFalse(
            Regex("""return %\w+ : tensor<1x4xf32>""").containsMatchIn(mlir),
            "function return type still 1x4 but the concat value is 1x8 — type mismatch:\n$mlir",
        )
    }

    @Test
    fun matmul_via_dag_dsl_declares_inferred_result_shape() {
        // (1,4)·(4,3) -> (1,3). Harness: OpsModel.matmulMlir().
        // dot_general contracts dim 1 x 0, so the result is 1x3 — but the export
        // declares the result/return as 1x4 (echoes operand-0), which iree-compile
        // rejects: "inferred shape '[1,3]' is incompatible with return type tensor<1x4xf32>".
        val mlir = lower("op_matmul") {
            val a = input<FP32>("a", TensorSpec("a", listOf(1, 4), "FP32"))
            val w = input<FP32>("w", TensorSpec("w", listOf(4, 3), "FP32"))
            output(matmul(a, w))
        }
        assertTrue(mlir.contains("stablehlo.dot_general") || mlir.contains("stablehlo.dot"), "matmul must lower:\n$mlir")
        assertTrue(mlir.contains("-> tensor<1x3xf32>"), "matmul result must be the inferred 1x3:\n$mlir")
        assertFalse(
            Regex("""return %\w+ : tensor<1x4xf32>""").containsMatchIn(mlir),
            "function return type still 1x4 but the matmul value is 1x3 — type mismatch:\n$mlir",
        )
    }
}
