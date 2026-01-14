package sk.ainet.lang.nn.compute

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

class Rgb2GrayScaleTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun makeVoidInput1x1(rgb: Triple<Float, Float, Float>): VoidOpsTensor<FP32, Float> {
        val (r, g, b) = rgb
        val data = dataFactory.fromFloatArray(
            shape = Shape(1, 3, 1, 1),
            data = floatArrayOf(r, g, b),
            dtype = FP32
        )
        return VoidOpsTensor(data, FP32::class)
    }

    private fun makeVoidInput1x2(
        r: Pair<Float, Float>,
        g: Pair<Float, Float>,
        b: Pair<Float, Float>
    ): Tensor<FP32, Float> {
        // NCHW layout: [R0, R1, G0, G1, B0, B1]
        val arr = floatArrayOf(r.first, r.second, g.first, g.second, b.first, b.second)
        val data = dataFactory.fromFloatArray(
            shape = Shape(1, 3, 1, 2),
            data = arr,
            dtype = FP32
        )
        return OpsBoundTensor.fromData(data, FP32::class, VoidTensorOps())
    }

    @Test
    fun testOutputShapeIsN1HW() {
        val ctx = DirectCpuExecutionContext()

        val model = Rgb2GrayScale()
        val module = model.model(ctx)

        val input = makeVoidInput1x2(
            r = 1.0f to 0.0f,
            g = 0.0f to 1.0f,
            b = 0.5f to 0.25f
        )

        val output = module.forward(input, ctx)
        assertEquals(Shape(1, 1, 1, 2), output.shape, "Output shape must be (N,1,H,W)")
        assertEquals(FP32::class, output.dtype)
    }

    @Test
    fun testZeroInputProducesZeroOutputShapeAndValues() {
        val ctx = DirectCpuExecutionContext()

        val model = Rgb2GrayScale()
        val module = model.model(ctx)

        val input = makeVoidInput1x1(Triple(0f, 0f, 0f))
        val output = module.forward(input, ctx)

        // Output shape should be (1,1,1,1) and value should be 0 with VoidOps
        assertEquals(Shape(1, 1, 1, 1), output.shape)
        val y = output.data[0, 0, 0, 0]
        assertEquals(0f, y)
    }
}
