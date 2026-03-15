package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

class TinyFoaOpsCpuTest {
    private val dataFactory = DenseTensorDataFactory()
    private val cpuOps = DefaultCpuOps(dataFactory)

    private fun fTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    private fun f16Tensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP16, Float> {
        val data = dataFactory.fromFloatArray<FP16, Float>(shape, FP16::class, values)
        return VoidOpsTensor(data, FP16::class)
    }

    private fun iTensor(shape: Shape, values: IntArray): VoidOpsTensor<Int32, Int> {
        val data = dataFactory.fromIntArray<Int32, Int>(shape, Int32::class, values)
        return VoidOpsTensor(data, Int32::class)
    }

    @Test
    fun testAbs() {
        // FP32
        val t1 = fTensor(Shape(3), floatArrayOf(-1f, 0f, 2f))
        val r1 = cpuOps.abs(t1)
        assertEquals(1f, r1.data[0])
        assertEquals(0f, r1.data[1])
        assertEquals(2f, r1.data[2])

        // FP16
        val t16 = f16Tensor(Shape(3), floatArrayOf(-1.5f, 0f, 2.5f))
        val r16 = cpuOps.abs(t16)
        assertEquals(1.5f, r16.data[0])
        assertEquals(0f, r16.data[1])
        assertEquals(2.5f, r16.data[2])

        // Int32
        val t2 = iTensor(Shape(3), intArrayOf(-5, 0, 10))
        val r2 = cpuOps.abs(t2)
        assertEquals(5, r2.data[0])
        assertEquals(0, r2.data[1])
        assertEquals(10, r2.data[2])
    }

    @Test
    fun testSign() {
        // FP32
        val t1 = fTensor(Shape(3), floatArrayOf(-1.5f, 0f, 2.5f))
        val r1 = cpuOps.sign(t1)
        assertEquals(-1f, r1.data[0])
        assertEquals(0f, r1.data[1])
        assertEquals(1f, r1.data[2])

        // Int32
        val t2 = iTensor(Shape(3), intArrayOf(-10, 0, 20))
        val r2 = cpuOps.sign(t2)
        assertEquals(-1, r2.data[0])
        assertEquals(0, r2.data[1])
        assertEquals(1, r2.data[2])
    }

    @Test
    fun testClamp() {
        // FP32
        val t1 = fTensor(Shape(3), floatArrayOf(-1f, 0.5f, 2f))
        val r1 = cpuOps.clamp(t1, 0f, 1f)
        assertEquals(0f, r1.data[0])
        assertEquals(0.5f, r1.data[1])
        assertEquals(1f, r1.data[2])

        // Int32
        val t2 = iTensor(Shape(3), intArrayOf(-5, 5, 15))
        val r2 = cpuOps.clamp(t2, 0f, 10f)
        assertEquals(0, r2.data[0])
        assertEquals(5, r2.data[1])
        assertEquals(10, r2.data[2])
    }

    @Test
    fun testLt() {
        // FP32
        val t1 = fTensor(Shape(3), floatArrayOf(1f, 2f, 3f))
        val r1 = cpuOps.lt(t1, 2.5f)
        assertEquals(1f, r1.data[0])
        assertEquals(1f, r1.data[1])
        assertEquals(0f, r1.data[2])

        // Int32
        val t2 = iTensor(Shape(3), intArrayOf(1, 2, 3))
        val r2 = cpuOps.lt(t2, 2f)
        assertEquals(1, r2.data[0])
        assertEquals(0, r2.data[1])
        assertEquals(0, r2.data[2])
    }

    @Test
    fun testGe() {
        // FP32
        val t1 = fTensor(Shape(3), floatArrayOf(1f, 2f, 3f))
        val r1 = cpuOps.ge(t1, 2f)
        assertEquals(0f, r1.data[0])
        assertEquals(1f, r1.data[1])
        assertEquals(1f, r1.data[2])

        // Int32
        val t2 = iTensor(Shape(3), intArrayOf(1, 2, 3))
        val r2 = cpuOps.ge(t2, 2f)
        assertEquals(0, r2.data[0])
        assertEquals(1, r2.data[1])
        assertEquals(1, r2.data[2])
    }

    @Test
    fun testNarrow() {
        val t1 = fTensor(Shape(2, 4), floatArrayOf(
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f
        ))
        
        // Narrow dim 1, start 1, length 2
        val r1 = cpuOps.narrow(t1, 1, 1, 2)
        assertEquals(Shape(2, 2), r1.shape)
        assertEquals(2f, r1.data[0, 0])
        assertEquals(3f, r1.data[0, 1])
        assertEquals(6f, r1.data[1, 0])
        assertEquals(7f, r1.data[1, 1])

        // Narrow dim 0, start 1, length 1
        val r2 = cpuOps.narrow(t1, 0, 1, 1)
        assertEquals(Shape(1, 4), r2.shape)
        assertEquals(5f, r2.data[0, 0])
        assertEquals(8f, r2.data[0, 3])
    }

    @Test
    fun testPad2d() {
        val t1 = fTensor(Shape(1, 1, 2, 2), floatArrayOf(
            1f, 2f,
            3f, 4f
        ))
        
        // Pad 1 all around
        val r1 = cpuOps.pad2d(t1, 1, 1, 1, 1)
        assertEquals(Shape(1, 1, 4, 4), r1.shape)
        
        // Check zeros
        assertEquals(0f, r1.data[0, 0, 0, 0])
        assertEquals(0f, r1.data[0, 0, 3, 3])
        
        // Check original data
        assertEquals(1f, r1.data[0, 0, 1, 1])
        assertEquals(2f, r1.data[0, 0, 1, 2])
        assertEquals(3f, r1.data[0, 0, 2, 1])
        assertEquals(4f, r1.data[0, 0, 2, 2])
    }

    @Test
    fun testUnfold() {
        val t1 = fTensor(Shape(1, 4), floatArrayOf(1f, 2f, 3f, 4f))
        
        // Unfold dim 1, size 2, step 1
        // (1, 4) -> (1, 3, 2)
        val r1 = cpuOps.unfold(t1, 1, 2, 1)
        assertEquals(Shape(1, 3, 2), r1.shape)
        
        // Window 0: [1, 2]
        assertEquals(1f, r1.data[0, 0, 0])
        assertEquals(2f, r1.data[0, 0, 1])
        // Window 1: [2, 3]
        assertEquals(2f, r1.data[0, 1, 0])
        assertEquals(3f, r1.data[0, 1, 1])
        // Window 2: [3, 4]
        assertEquals(3f, r1.data[0, 2, 0])
        assertEquals(4f, r1.data[0, 2, 1])

        // Unfold dim 1, size 2, step 2
        // (1, 4) -> (1, 2, 2)
        val r2 = cpuOps.unfold(t1, 1, 2, 2)
        assertEquals(Shape(1, 2, 2), r2.shape)
        // Window 0: [1, 2]
        assertEquals(1f, r2.data[0, 0, 0])
        assertEquals(2f, r2.data[0, 0, 1])
        // Window 1: [3, 4]
        assertEquals(3f, r2.data[0, 1, 0])
        assertEquals(4f, r2.data[0, 1, 1])
    }
}
