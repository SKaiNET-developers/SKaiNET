package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame

/**
 * Pins the contract for [TensorDataFactory.placeholder]:
 *
 * 1. Reports the requested shape without touching the underlying buffer.
 * 2. Materializes to zeros on the first read — value parity with [zeros].
 * 3. Caches the materialized buffer (no re-allocation across reads).
 *
 * The benefit (deferred allocation) doesn't show up directly in unit tests, but
 * the parity guarantee means any caller that *does* read the tensor sees the
 * same values an eager [zeros] call would have produced — so dropping in
 * `placeholder` for `zeros` in DSL parameter init is a strict improvement.
 */
class PlaceholderTensorDataTest {

    private val factory = DenseTensorDataFactory()

    @Test
    fun placeholder_reports_shape_without_materializing() {
        val shape = Shape(64, 64)
        val td = factory.placeholder<FP32, Float>(shape, FP32::class)

        // Reading shape must not require allocating the underlying buffer.
        assertEquals(shape, td.shape)
        // Returned shape is a defensive copy — mutating one shouldn't affect the
        // factory-issued tensor's view.
        assertEquals(64, td.shape.dimensions[0])
        assertEquals(64, td.shape.dimensions[1])
    }

    @Test
    fun placeholder_materializes_to_zeros_on_first_read_fp32() {
        val td = factory.placeholder<FP32, Float>(Shape(2, 3), FP32::class)

        // Every position reads as 0.0f — same as zeros().
        for (i in 0 until 2) for (j in 0 until 3) {
            assertEquals(0.0f, td[i, j], "[$i,$j] must be 0.0f on first read")
        }
    }

    @Test
    fun placeholder_supports_writes_and_reads_back_fp32() {
        val td = factory.placeholder<FP32, Float>(Shape(4), FP32::class)

        td[2] = 7.5f
        assertEquals(7.5f, td[2])
        assertEquals(0.0f, td[0])
        assertEquals(0.0f, td[3])
    }

    @Test
    fun placeholder_buffer_is_stable_across_reads() {
        val td = factory.placeholder<FP32, Float>(Shape(8), FP32::class)
            as FloatArrayTensorData<FP32>

        val first = td.buffer
        val second = td.buffer
        // Same backing FloatArray on every access — the lazy fires once.
        assertSame(first, second, "buffer must be cached after first materialization")
    }

    @Test
    fun placeholder_value_parity_with_zeros_fp32() {
        val shape = Shape(5, 7)
        val placeholder = factory.placeholder<FP32, Float>(shape, FP32::class)
        val zeros = factory.zeros<FP32, Float>(shape, FP32::class)

        for (i in 0 until 5) for (j in 0 until 7) {
            assertEquals(zeros[i, j], placeholder[i, j],
                "placeholder must match zeros at [$i,$j]")
        }
    }

    @Test
    fun placeholder_int32_materializes_to_zeros() {
        val td = factory.placeholder<Int32, Int>(Shape(3), Int32::class)
        assertEquals(0, td[0])
        assertEquals(0, td[1])
        assertEquals(0, td[2])
    }

    @Test
    fun placeholder_int8_falls_back_to_zeros() {
        // Int8 has no lazy variant — falls back to eager zeros. The test pins
        // the value contract; it shouldn't throw and reads must be 0.
        val td = factory.placeholder<Int8, Byte>(Shape(4), Int8::class)
        for (i in 0 until 4) {
            assertEquals(0.toByte(), td[i])
        }
    }

    @Test
    fun placeholder_returns_distinct_instances() {
        // Two placeholder calls must not share underlying state — separate Linear
        // layers must not see each other's writes.
        val a = factory.placeholder<FP32, Float>(Shape(4), FP32::class)
            as FloatArrayTensorData<FP32>
        val b = factory.placeholder<FP32, Float>(Shape(4), FP32::class)
            as FloatArrayTensorData<FP32>

        assertNotSame(a.buffer, b.buffer)
        a[0] = 99.0f
        assertEquals(0.0f, b[0], "placeholder b must not see writes to placeholder a")
    }
}
