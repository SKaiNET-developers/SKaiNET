package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.storage.ActiveMemoryTracker
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.types.FP32
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * #1247 regression gate: shape-only (void) tracing must not materialize data.
 *
 * Before the lazy-placeholder switch, every one of VoidTensorOps' ~62
 * shape-propagating ops allocated a real dense zeros buffer — twice
 * transiently, once retained — so a 30-layer Gemma 3n E2B trace retained
 * ~9 GB of zeros (worst case a weight-sized buffer per projection via
 * matmulWeightTransposed's transpose). These tests pin the new contract:
 * weight-scale void ops record zero tracked allocation, and reading an
 * element of a static void tensor still observes the historical zeros
 * (lazily materialized).
 */
class VoidOpsAllocationTest {

    private val ops = VoidTensorOps()
    private val dense = DenseTensorDataFactory()

    @AfterTest
    fun teardown() {
        ActiveMemoryTracker.current = null
    }

    private fun voidTensor(vararg dims: Int): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(dense.placeholder(Shape(dims), FP32::class), FP32::class)

    @Test
    fun weight_scale_void_op_stack_allocates_nothing() {
        val x = voidTensor(1, 4096)
        val weight = voidTensor(4096, 4096) // 64 MiB if it were dense FP32
        val bias = voidTensor(1, 4096)

        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        // A projection stack the tracer records per transformer layer.
        val projected = ops.matmulWeightTransposed(x, weight)
        val transposed = ops.transpose(weight)
        val product = ops.matmul(x, transposed)
        val summed = ops.add(product, bias)
        val activated = ops.relu(summed)

        // Shapes propagate…
        assertEquals(Shape(1, 4096), projected.shape)
        assertEquals(Shape(4096, 4096), transposed.shape)
        assertEquals(Shape(1, 4096), product.shape)
        assertEquals(Shape(1, 4096), summed.shape)
        assertEquals(Shape(1, 4096), activated.shape)

        // …and nothing materializes: no zeros buffers, no copies.
        val report = tracker.report()
        assertEquals(
            0L, report.copyBytes,
            "shape-only tracing must not allocate; tracked ${report.copyBytes} bytes " +
                "across ${report.copyCount} copies: ${report.copiesBySource}"
        )
    }

    @Test
    fun reading_a_static_void_tensor_still_yields_zeros() {
        // Compat contract for the pre-#1247 readers of void zeros: the value
        // is still 0.0f, just materialized lazily on first access.
        val result = ops.add(voidTensor(2, 3), voidTensor(2, 3))
        assertEquals(0.0f, result.data.get(1, 2))
    }

    @Test
    fun matmulWeightTransposed_override_matches_default_shape_semantics() {
        // [batch, in] x [out, in] -> [batch, out], identical to
        // matmul(x, transpose(weight)) without the intermediate.
        val viaOverride = ops.matmulWeightTransposed(voidTensor(3, 8), voidTensor(16, 8))
        val viaDefault = ops.matmul(voidTensor(3, 8), ops.transpose(voidTensor(16, 8)))
        assertEquals(viaDefault.shape, viaOverride.shape)
        assertEquals(Shape(3, 16), viaOverride.shape)
    }
}
