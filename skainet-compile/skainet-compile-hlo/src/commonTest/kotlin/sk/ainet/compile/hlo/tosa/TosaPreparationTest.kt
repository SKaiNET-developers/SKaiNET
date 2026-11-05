package sk.ainet.compile.hlo.tosa

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.applyPasses
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.compile.opt.StableHloPass
import sk.ainet.compile.opt.TargetOptimizers
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class TosaPreparationTest {
    @AfterTest
    fun reset() = TargetOptimizers.clear()

    // a layer norm's tail and a softmax normalize, as the emitter writes them
    private val divides = """
        func.func @f(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x1xf32>, %arg2: tensor<1x4x1xf32>) -> tensor<1x4x8xf32> {
          %v1 = stablehlo.sqrt %arg1 : tensor<1x4x1xf32>
          %v2 = stablehlo.broadcast_in_dim %v1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %v3 = stablehlo.divide %arg0, %v2 : tensor<1x4x8xf32>
          %v4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %v5 = stablehlo.divide %v3, %v4 : tensor<1x4x8xf32>
          return %v5 : tensor<1x4x8xf32>
        }
    """.trimIndent()

    @Test
    fun a_divide_by_a_broadcast_becomes_a_multiply_by_the_broadcast_reciprocal_rsqrt_for_a_sqrt() {
        val (out, n) = DivideByBroadcastPass.rewrite(divides)
        assertEquals(2, n)
        assertContains(out, "%tq_r1 = stablehlo.rsqrt %arg1 : tensor<1x4x1xf32>")
        assertContains(out, "%tq_b1 = stablehlo.broadcast_in_dim %tq_r1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>")
        assertContains(out, "%v3 = stablehlo.multiply %arg0, %tq_b1 : tensor<1x4x8xf32>")
        assertContains(out, "%tq_r2 = stablehlo.divide %tq_one2, %arg2 : tensor<1x4x1xf32>")
        assertContains(out, "%v5 = stablehlo.multiply %v3, %tq_b2 : tensor<1x4x8xf32>")
    }

    @Test
    fun a_divide_by_an_unbroadcast_value_is_left_alone() {
        val mlir = "    %v1 = stablehlo.divide %arg0, %arg1 : tensor<1x4x8xf32>"
        assertEquals(mlir to 0, DivideByBroadcastPass.rewrite(mlir))
    }

    private val matmuls = """
        func.func @f(%arg0: tensor<1x4x8xf32>, %w: tensor<8x6xf32>, %q: tensor<1x2x4x3xf32>, %k: tensor<1x2x4x3xf32>) -> tensor<1x4x6xf32> {
          %v1 = stablehlo.dot_general %arg0, %w, contracting_dims = [2] x [0] : (tensor<1x4x8xf32>, tensor<8x6xf32>) -> tensor<1x4x6xf32>
          %v2 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x4x3xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x4xf32>
          return %v1 : tensor<1x4x6xf32>
        }
    """.trimIndent()

    @Test
    fun a_weight_matmul_becomes_a_rank_two_dot_the_legalization_converts_attention_is_left_alone() {
        val (out, n) = RankTwoWeightMatmulPass.rewrite(matmuls)
        assertEquals(1, n)
        assertContains(out, "%tq_mm1_a = stablehlo.reshape %arg0 : (tensor<1x4x8xf32>) -> tensor<4x8xf32>")
        assertContains(out, "%tq_mm1_d = stablehlo.dot_general %tq_mm1_a, %w, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>")
        assertContains(out, "%v1 = stablehlo.reshape %tq_mm1_d : (tensor<4x6xf32>) -> tensor<1x4x6xf32>")
        assertContains(out, "%v2 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1]")
    }

    @Test
    fun passes_registered_for_a_target_apply_in_order_through_the_module() {
        TargetOptimizers.registerStableHloPasses("npu") { TosaPreparation.passes() }
        val passes = TargetOptimizers.stableHloPassesFor("npu")
        assertEquals(listOf("tosa-divide-by-broadcast", "tosa-rank-two-weight-matmul"), passes.map { it.name })
        assertTrue(TargetOptimizers.stableHloPassesFor("llvm-cpu").isEmpty())
        assertTrue(TargetOptimizers.stableHloPassesFor(null).isEmpty())

        val module = StableHloModule(divides + "\n" + matmuls)
        val out = module.applyPasses(passes).content
        assertFalse("stablehlo.divide %arg0, %v2" in out)
        assertContains(out, "%tq_mm1_d = stablehlo.dot_general")
        assertSame(module, module.applyPasses(emptyList()))
    }

    @Test
    fun several_optimizers_for_one_target_contribute_their_passes_in_registration_order() {
        val tag = object : StableHloPass {
            override val name = "tag"
            override fun apply(mlir: String) = "$mlir\n// tagged"
        }
        TargetOptimizers.registerStableHloPasses("npu") { listOf(DivideByBroadcastPass) }
        TargetOptimizers.registerStableHloPasses("npu") { listOf(tag) }
        val out = StableHloModule(divides).applyPasses(TargetOptimizers.stableHloPassesFor("npu")).content
        assertTrue(out.endsWith("// tagged"))
        assertContains(out, "stablehlo.multiply %arg0, %tq_b1")
    }

    @Test
    fun the_emitter_applies_a_targets_passes_to_the_module_it_emits() {
        val graph = DefaultComputeGraph()
        val spec = TensorSpec("x", listOf(1, 4), "FP32")
        val x = GraphNode("x", InputOperation<DType, Any>(), emptyList(), listOf(spec))
        val add = GraphNode("add", AddOperation<DType, Any>(), listOf(spec, spec), listOf(TensorSpec("y", listOf(1, 4), "FP32")))
        graph.addNode(x); graph.addNode(add)
        graph.addEdge(GraphEdge("e0", x, add, 0, 0, spec))
        graph.addEdge(GraphEdge("e1", x, add, 0, 1, spec))
        val tag = object : StableHloPass {
            override val name = "tag"
            override fun apply(mlir: String) = "$mlir\n// tagged"
        }
        TargetOptimizers.registerStableHloPasses("npu") { listOf(tag) }

        val plain = toStableHlo(graph, "f", target = "npu").content
        val passed = toStableHlo(graph, "f", target = "npu", passes = TargetOptimizers.stableHloPassesFor("npu")).content
        assertFalse(plain.endsWith("// tagged"))
        assertEquals("$plain\n// tagged", passed)
    }

    // what --stablehlo-legalize-to-tosa leaves: broadcasts, an rsqrt, a weight matmul and attention's two batched ones
    private val legalized = """
        func.func @f(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x1xf32>, %arg2: tensor<8x6xf32>, %q: tensor<1x2x4x3xf32>, %k: tensor<1x2x4x3xf32>, %p: tensor<1x2x4x4xf32>) -> tensor<1x4x8xf32> {
          %0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
          %1 = stablehlo.rsqrt %arg1 : tensor<1x4x1xf32>
          %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %3 = tosa.mul %arg0, %2, %0 : (tensor<1x4x8xf32>, tensor<1x4x8xf32>, tensor<1xi8>) -> tensor<1x4x8xf32>
          %4 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
          %5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<1x4x8xf32>) -> tensor<1x4x1xf32>
          %6 = stablehlo.dot_general %3, %arg2, contracting_dims = [2] x [0] : (tensor<1x4x8xf32>, tensor<8x6xf32>) -> tensor<1x4x6xf32>
          %7 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<1x2x4x3xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x4xf32>
          %8 = stablehlo.dot_general %p, %q, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x2x4x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
          return %3 : tensor<1x4x8xf32>
        }
    """.trimIndent()

    @Test
    fun broadcasts_feeding_elementwise_ops_stay_rank_equal_other_broadcasts_tile() {
        val r = TosaFinish.finish(legalized)
        assertEquals(2, r.broadcasts); assertEquals(1, r.tiled); assertEquals(1, r.roots)
        assertContains(r.mlir, "%1 = tosa.rsqrt %arg1 : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>")
        // %2 only feeds the mul: no tile, the mul reads the [1,4,1] value and broadcasts it itself
        assertContains(r.mlir, "%tq_bc2 = tosa.identity %1 : (tensor<1x4x1xf32>) -> tensor<1x4x1xf32>")
        assertContains(r.mlir, "%3 = tosa.mul %arg0, %tq_bc2, %0 : (tensor<1x4x8xf32>, tensor<1x4x1xf32>, tensor<1xi8>) -> tensor<1x4x8xf32>")
        // %4 feeds a reduce: tiled to the full shape
        assertContains(r.mlir, "values = dense<[1, 1, 8]> : tensor<3xindex>")
        assertContains(r.mlir, "%4 = tosa.tile %arg1, ")
        assertFalse("stablehlo." in r.mlir, "nothing of StableHLO is left")
    }

    @Test
    fun dot_generals_become_tosa_matmuls_a_weight_reshaped_to_batch_1_attention_per_head() {
        val r = TosaFinish.finish(legalized)
        assertEquals(1, r.weightMatmuls); assertEquals(2, r.batchedMatmuls)
        assertContains(r.mlir, "%tq_zp = \"tosa.const\"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>")
        assertContains(r.mlir, "-> tensor<1x8x6xf32>")
        assertContains(r.mlir, "%6 = tosa.matmul %3, %tq_rs")
        assertContains(r.mlir, ": (tensor<1x4x8xf32>, tensor<1x8x6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x4x6xf32>")
        // q · kᵀ: k transposed in its last two dimensions, then [2,4,3] x [2,3,4]
        assertContains(r.mlir, "tosa.transpose %k {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x4x3xf32>) -> tensor<1x2x3x4xf32>")
        assertContains(r.mlir, ": (tensor<2x4x3xf32>, tensor<2x3x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x4xf32>")
        assertContains(r.mlir, "%7 = tosa.reshape %tq_bm")
        // p · v: [2,4,4] x [2,4,3]
        assertContains(r.mlir, ": (tensor<2x4x4xf32>, tensor<2x4x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x3xf32>")
    }

    @Test
    fun a_stablehlo_op_the_translation_does_not_handle_fails_it() {
        val e = assertFailsWith<TosaFinishException> {
            TosaFinish.finish(
                legalized.replace(
                    "%5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<1x4x8xf32>) -> tensor<1x4x1xf32>",
                    "%5 = stablehlo.cosine %4 : tensor<1x4x8xf32>",
                ),
            )
        }
        assertContains(e.message!!, "the legalization left `stablehlo.cosine`")
    }
}
