package sk.ainet.lang.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class TinyFoaOpsVoidTest {
    private val dataFactory = DenseTensorDataFactory()
    private val voidOps = VoidTensorOps()

    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun testAbs_ShapeAndDtype() {
        val t = createTensor(Shape(2, 3))
        val r = voidOps.abs(t)
        assertEquals(Shape(2, 3), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testSign_ShapeAndDtype() {
        val t = createTensor(Shape(2, 3))
        val r = voidOps.sign(t)
        assertEquals(Shape(2, 3), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testClamp_ShapeAndDtype() {
        val t = createTensor(Shape(2, 3))
        val r = voidOps.clamp(t, 0f, 1f)
        assertEquals(Shape(2, 3), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testLt_ShapeAndDtype() {
        val t = createTensor(Shape(2, 3))
        val r = voidOps.lt(t, 0.5f)
        assertEquals(Shape(2, 3), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testGe_ShapeAndDtype() {
        val t = createTensor(Shape(2, 3))
        val r = voidOps.ge(t, 0.5f)
        assertEquals(Shape(2, 3), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testNarrow_ShapeAndDtype() {
        val t = createTensor(Shape(2, 4))
        val r = voidOps.narrow(t, 1, 1, 2)
        assertEquals(Shape(2, 2), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testPad2d_ShapeAndDtype() {
        val t = createTensor(Shape(1, 1, 2, 2))
        val r = voidOps.pad2d(t, 1, 1, 1, 1)
        assertEquals(Shape(1, 1, 4, 4), r.shape)
        assertEquals(FP32::class, r.dtype)
    }

    @Test
    fun testUnfold_ShapeAndDtype() {
        val t = createTensor(Shape(1, 4))
        val r = voidOps.unfold(t, 1, 2, 1)
        assertEquals(Shape(1, 3, 2), r.shape)
        assertEquals(FP32::class, r.dtype)
    }
}
