package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertNotSame
import kotlin.test.assertTrue

class ExplicitCopyApiTest {

    private val factory = DenseTensorDataFactory()

    // --- wrapFloatArray (zero-copy) ---

    @Test
    fun wrapFloatArraySharesBuffer() {
        val original = floatArrayOf(1f, 2f, 3f, 4f)
        val wrapped = factory.wrapFloatArray<FP32, Float>(Shape(4), FP32::class, original)

        assertTrue(wrapped is FloatArrayTensorData<*>)
        val floatData = wrapped as FloatArrayTensorData<*>
        // The buffer IS the same array (zero-copy)
        assertSame(original, floatData.buffer)
    }

    @Test
    fun fromFloatArrayCopiesBuffer() {
        val original = floatArrayOf(1f, 2f, 3f, 4f)
        val copied = factory.fromFloatArray<FP32, Float>(Shape(4), FP32::class, original)

        assertTrue(copied is FloatArrayTensorData<*>)
        val floatData = copied as FloatArrayTensorData<*>
        // The buffer is a DIFFERENT array (copy)
        assertNotSame(original, floatData.buffer)
        // But same contents
        assertEquals(original.toList(), floatData.buffer.toList())
    }

    @Test
    fun wrapFloatArrayMutationsVisibleThroughTensorData() {
        val original = floatArrayOf(10f, 20f, 30f)
        val wrapped = factory.wrapFloatArray<FP32, Float>(Shape(3), FP32::class, original)

        // Mutate original
        original[0] = 99f
        // Change is visible through the wrapped tensor data
        assertEquals(99f, wrapped[0])
    }

    // --- wrapIntArray (zero-copy) ---

    @Test
    fun wrapIntArraySharesBuffer() {
        val original = intArrayOf(10, 20, 30)
        val wrapped = factory.wrapIntArray<Int32, Int>(Shape(3), Int32::class, original)

        val intData = wrapped as sk.ainet.lang.tensor.data.IntArrayTensorData<*>
        assertSame(original, intData.buffer)
    }

    @Test
    fun fromIntArrayCopiesBuffer() {
        val original = intArrayOf(10, 20, 30)
        val copied = factory.fromIntArray<Int32, Int>(Shape(3), Int32::class, original)

        val intData = copied as sk.ainet.lang.tensor.data.IntArrayTensorData<*>
        assertNotSame(original, intData.buffer)
    }

    // --- TensorStorage bridge with borrowed vs owned ---

    @Test
    fun tensorStorageFromBorrowedRawBytes() {
        val rawData = ByteArray(144) // Q4_K block
        val storage = TensorStorageFactory.fromRawBytes(
            shape = Shape(256),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            data = rawData
        )
        assertEquals(Ownership.BORROWED, storage.ownership)
    }

    @Test
    fun tensorStorageFromOwnedRawBytes() {
        val rawData = ByteArray(144)
        val storage = TensorStorageFactory.fromRawBytesOwned(
            shape = Shape(256),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            data = rawData
        )
        assertEquals(Ownership.OWNED, storage.ownership)
    }
}
