package sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
    fun rejects_Q_and_K_with_mismatched_head_count() {
        val q = zeros(Shape(1, 8, 1, 64))
        val k = zeros(Shape(1, 4, 1, 64)) // K has fewer heads — ungrouped K/V tiling never happened
        val v = zeros(Shape(1, 4, 1, 64))
        assertFailsWith<IllegalArgumentException> {
            ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 1f, causal = true)
        }
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
}
