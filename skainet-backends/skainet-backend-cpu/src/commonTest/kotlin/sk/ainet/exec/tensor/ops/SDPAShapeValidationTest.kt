package sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Shape-validation tests for [sk.ainet.lang.tensor.ops.TensorOps.scaledDotProductAttention].
 *
 * Pins the precondition contract: Q/K/V must agree on `batch`, `heads`,
 * `headDim`, and K/V must agree on `seqKV`. Without these, a mismatched
 * head_dim surfaces as an `ArrayIndexOutOfBoundsException` thousands of
 * lines from the caller — that's the exact symptom we hit on real
 * Gemma 4 E2B (mixed-head_dim shared KV: Q comes in at headDim=512 while
 * K comes back at headDim=256 because the shared owner cache was sized
 * for the owner layer's SLIDING head_dim).
 */
class SDPAShapeValidationTest {

    private val ctx = DirectCpuExecutionContext()

    private fun zeros(shape: Shape): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume))

    @Test
    fun rejects_Q_and_K_with_mismatched_head_dim() {
        // Q: [batch=1, heads=8, seqQ=1, headDim=512]  (GLOBAL-layer shape)
        // K: [batch=1, heads=8, seqKV=1, headDim=256] (shared-owner SLIDING shape)
        val q = zeros(Shape(1, 8, 1, 512))
        val k = zeros(Shape(1, 8, 1, 256))
        val v = zeros(Shape(1, 8, 1, 256))
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        }
    }

    @Test
    fun rejects_Q_and_V_with_mismatched_head_dim() {
        val q = zeros(Shape(1, 4, 2, 8))
        val k = zeros(Shape(1, 4, 2, 8))
        val v = zeros(Shape(1, 4, 2, 16))
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        }
    }

    @Test
    fun rejects_Q_and_K_with_nonDividing_head_count() {
        // SKEEP-005 phase 2: grouped-query attention is native — K/V heads that DIVIDE Q heads
        // are the contract; a count that does not divide is still a caller bug.
        val q = zeros(Shape(1, 6, 1, 64))
        val k = zeros(Shape(1, 4, 1, 64))
        val v = zeros(Shape(1, 4, 1, 64))
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        }
    }

    @Test
    fun accepts_grouped_query_head_counts() {
        val q = zeros(Shape(1, 8, 1, 64))
        val k = zeros(Shape(1, 4, 1, 64))
        val v = zeros(Shape(1, 4, 1, 64))
        val out = ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        kotlin.test.assertEquals(listOf(1, 8, 1, 64), out.shape.dimensions.toList())
    }

    @Test
    fun rejects_K_and_V_with_mismatched_seqKV() {
        val q = zeros(Shape(1, 2, 1, 8))
        val k = zeros(Shape(1, 2, 4, 8))
        val v = zeros(Shape(1, 2, 5, 8))
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        }
    }

    @Test
    fun accepts_matching_shapes_and_returns_Q_shape_output() {
        // Happy path: Q=K=V shapes agree on batch, heads, headDim; seqQ==seqKV.
        val q = zeros(Shape(1, 2, 3, 4))
        val k = zeros(Shape(1, 2, 3, 4))
        val v = zeros(Shape(1, 2, 3, 4))
        val out = ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        assertEquals(Shape(1, 2, 3, 4), out.shape)
    }

    @Test
    fun default_scale_uses_one_over_sqrt_head_dim_not_zero() {
        // Regression for #860: the default scale = 0f must be resolved to
        // 1/sqrt(headDim), not applied literally (which flattens the softmax
        // to a uniform average and silently discards the attention pattern).
        val rng = kotlin.random.Random(42)
        fun rnd(shape: Shape) = ctx.fromFloatArray<FP32, Float>(shape, FP32::class, FloatArray(shape.volume) { rng.nextFloat() })
        val headDim = 4
        val q = rnd(Shape(1, 1, 3, headDim))
        val k = rnd(Shape(1, 1, 3, headDim))
        val v = rnd(Shape(1, 1, 3, headDim))

        val defaulted = ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 0f, causal = true)
        val explicit = ctx.ops.scaledDotProductAttention(
            q, k, v, mask = null, scale = (1.0 / kotlin.math.sqrt(headDim.toDouble())).toFloat(), causal = true,
        )
        // The two must agree...
        val a = defaulted.data.copyToFloatArray()
        val b = explicit.data.copyToFloatArray()
        for (i in a.indices) assertEquals(b[i], a[i], 1e-6f)

        // ...and must differ from the degenerate scale=0 (uniform) result.
        val uniform = ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = Float.MIN_VALUE, causal = true)
        val u = uniform.data.copyToFloatArray()
        assertTrue(a.indices.any { kotlin.math.abs(a[it] - u[it]) > 1e-4f }, "default scale must not equal the near-zero (uniform) result")
    }
}
