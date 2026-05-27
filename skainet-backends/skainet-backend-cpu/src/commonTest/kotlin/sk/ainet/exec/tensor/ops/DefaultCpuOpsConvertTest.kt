package sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32

class DefaultCpuOpsConvertTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOps(dataFactory)

    private fun fp32Tensor(
        shape: Shape,
        values: FloatArray,
        requiresGrad: Boolean = false
    ): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class, GradState(requiresGrad = requiresGrad))
    }

    private fun int32Tensor(shape: Shape, values: IntArray): VoidOpsTensor<Int32, Int> {
        val data = dataFactory.fromIntArray<Int32, Int>(shape, Int32::class, values)
        return VoidOpsTensor(data, Int32::class)
    }

    @Test
    fun convertFp32ToFp16PreservesShapeValuesAndGradRequirement() {
        val input = fp32Tensor(
            Shape(2, 2),
            floatArrayOf(1.25f, -2.5f, 3.75f, 4.5f),
            requiresGrad = true
        )

        val result = ops.convert(input, FP16)

        assertEquals(Shape(2, 2), result.shape)
        assertEquals(FP16::class, result.dtype)
        assertTrue(result.requiresGrad)
        assertContentEquals(
            floatArrayOf(1.25f, -2.5f, 3.75f, 4.5f),
            (result.data as FloatArrayTensorData<*>).buffer
        )
    }

    @Test
    fun convertInt32ToFp32CastsValuesToFloat() {
        val input = int32Tensor(Shape(2, 2), intArrayOf(1, -2, 3, 4))

        val result = ops.convert(input, FP32)

        assertEquals(Shape(2, 2), result.shape)
        assertEquals(FP32::class, result.dtype)
        assertContentEquals(
            floatArrayOf(1f, -2f, 3f, 4f),
            (result.data as FloatArrayTensorData<*>).buffer
        )
    }

    @Test
    fun convertFp32ToInt32CastsValuesToInt() {
        val input = fp32Tensor(Shape(4), floatArrayOf(1.9f, -2.1f, 3.0f, 4.8f))

        val result = ops.convert(input, Int32)

        assertEquals(Shape(4), result.shape)
        assertEquals(Int32::class, result.dtype)
        assertContentEquals(intArrayOf(1, -2, 3, 4), (result.data as IntArrayTensorData<*>).buffer)
    }

    @Test
    fun convertToSameDtypeReturnsInputTensor() {
        val input = fp32Tensor(Shape(2), floatArrayOf(1f, 2f))

        val result = ops.convert(input, FP32)

        assertSame(input, result)
    }

    @Test
    fun convertRejectsUnsupportedTargetDtype() {
        val input = fp32Tensor(Shape(2), floatArrayOf(1f, 2f))

        assertFailsWith<IllegalArgumentException> {
            ops.convert(input, Int16)
        }
    }
}
