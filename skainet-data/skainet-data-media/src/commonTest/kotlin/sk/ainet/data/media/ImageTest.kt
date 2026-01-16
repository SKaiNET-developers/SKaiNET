package sk.ainet.data.media

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith

class ImageTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    // ========== HWC Layout Tests ==========

    @Test
    fun testHWCImage() {
        val tensor = createTensor(Shape(480, 640, 3)) // height, width, channels
        val image = Image.fromTensor(tensor, ImageLayout.HWC, ColorSpace.RGB)

        assertEquals(640, image.width)
        assertEquals(480, image.height)
        assertEquals(3, image.channels)
        assertEquals(1, image.batchSize)
        assertFalse(image.isBatched)
        assertEquals(480 * 640, image.pixelCount)
        assertTrue(image.isConsistent)
    }

    @Test
    fun testHWCGrayscale() {
        val tensor = createTensor(Shape(100, 200, 1))
        val image = Image.fromTensor(tensor, ImageLayout.HWC, ColorSpace.GRAYSCALE)

        assertEquals(200, image.width)
        assertEquals(100, image.height)
        assertEquals(1, image.channels)
        assertTrue(image.isConsistent)
    }

    @Test
    fun testHWCRGBA() {
        val tensor = createTensor(Shape(256, 256, 4))
        val image = Image.fromTensor(tensor, ImageLayout.HWC, ColorSpace.RGBA)

        assertEquals(256, image.width)
        assertEquals(256, image.height)
        assertEquals(4, image.channels)
        assertTrue(image.isConsistent)
    }

    // ========== CHW Layout Tests ==========

    @Test
    fun testCHWImage() {
        val tensor = createTensor(Shape(3, 480, 640)) // channels, height, width
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        assertEquals(640, image.width)
        assertEquals(480, image.height)
        assertEquals(3, image.channels)
        assertEquals(1, image.batchSize)
        assertFalse(image.isBatched)
    }

    @Test
    fun testCHWGrayscale() {
        val tensor = createTensor(Shape(1, 28, 28)) // MNIST-like
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.GRAYSCALE)

        assertEquals(28, image.width)
        assertEquals(28, image.height)
        assertEquals(1, image.channels)
        assertTrue(image.isConsistent)
    }

    // ========== NHWC Layout Tests ==========

    @Test
    fun testNHWCBatchedImage() {
        val tensor = createTensor(Shape(32, 224, 224, 3)) // batch, height, width, channels
        val image = Image.fromTensor(tensor, ImageLayout.NHWC, ColorSpace.RGB)

        assertEquals(224, image.width)
        assertEquals(224, image.height)
        assertEquals(3, image.channels)
        assertEquals(32, image.batchSize)
        assertTrue(image.isBatched)
    }

    // ========== NCHW Layout Tests ==========

    @Test
    fun testNCHWBatchedImage() {
        val tensor = createTensor(Shape(16, 3, 299, 299)) // batch, channels, height, width
        val image = Image.fromTensor(tensor, ImageLayout.NCHW, ColorSpace.RGB)

        assertEquals(299, image.width)
        assertEquals(299, image.height)
        assertEquals(3, image.channels)
        assertEquals(16, image.batchSize)
        assertTrue(image.isBatched)
    }

    @Test
    fun testNCHWSingleBatch() {
        val tensor = createTensor(Shape(1, 3, 512, 512))
        val image = Image.fromTensor(tensor, ImageLayout.NCHW, ColorSpace.RGB)

        assertEquals(512, image.width)
        assertEquals(512, image.height)
        assertEquals(1, image.batchSize)
        assertTrue(image.isBatched) // Still batched, just with batch size 1
    }

    // ========== Color Space Inference Tests ==========

    @Test
    fun testInferGrayscale() {
        val tensor = createTensor(Shape(100, 100, 1))
        val image = Image.fromTensor(tensor, ImageLayout.HWC)

        assertEquals(ColorSpace.GRAYSCALE, image.colorSpace)
        assertEquals(1, image.channels)
    }

    @Test
    fun testInferRGB() {
        val tensor = createTensor(Shape(100, 100, 3))
        val image = Image.fromTensor(tensor, ImageLayout.HWC)

        assertEquals(ColorSpace.RGB, image.colorSpace)
        assertEquals(3, image.channels)
    }

    @Test
    fun testInferRGBA() {
        val tensor = createTensor(Shape(100, 100, 4))
        val image = Image.fromTensor(tensor, ImageLayout.HWC)

        assertEquals(ColorSpace.RGBA, image.colorSpace)
        assertEquals(4, image.channels)
    }

    @Test
    fun testInferColorSpaceFailsForUnknownChannels() {
        val tensor = createTensor(Shape(100, 100, 5))

        assertFailsWith<IllegalArgumentException> {
            Image.fromTensor(tensor, ImageLayout.HWC)
        }
    }

    // ========== Consistency Tests ==========

    @Test
    fun testInconsistentChannels() {
        val tensor = createTensor(Shape(100, 100, 3))
        val image = Image.fromTensor(tensor, ImageLayout.HWC, ColorSpace.GRAYSCALE)

        assertFalse(image.isConsistent) // 3 channels but GRAYSCALE expects 1
    }

    @Test
    fun testConsistentBGR() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.BGR)

        assertTrue(image.isConsistent) // 3 channels, BGR expects 3
    }

    // ========== withLayout Tests ==========

    @Test
    fun testWithLayoutSameRank() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        // HWC also has rank 3, so metadata change is allowed
        val reinterpreted = image.withLayout(ImageLayout.HWC)

        assertEquals(ImageLayout.HWC, reinterpreted.layout)
        assertEquals(ColorSpace.RGB, reinterpreted.colorSpace)
        // Note: dimensions will be "wrong" because we're just reinterpreting
    }

    @Test
    fun testWithLayoutDifferentRankFails() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        assertFailsWith<IllegalArgumentException> {
            image.withLayout(ImageLayout.NCHW) // 4D layout, but tensor is 3D
        }
    }

    @Test
    fun testWithLayoutBatchedToBatched() {
        val tensor = createTensor(Shape(8, 3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.NCHW, ColorSpace.RGB)

        val reinterpreted = image.withLayout(ImageLayout.NHWC)

        assertEquals(ImageLayout.NHWC, reinterpreted.layout)
    }

    // ========== withColorSpace Tests ==========

    @Test
    fun testWithColorSpaceSameChannels() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        val reinterpreted = image.withColorSpace(ColorSpace.BGR)

        assertEquals(ColorSpace.BGR, reinterpreted.colorSpace)
        assertEquals(ImageLayout.CHW, reinterpreted.layout)
    }

    @Test
    fun testWithColorSpaceYUV() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        val reinterpreted = image.withColorSpace(ColorSpace.YUV)

        assertEquals(ColorSpace.YUV, reinterpreted.colorSpace)
    }

    @Test
    fun testWithColorSpaceDifferentChannelsFails() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        assertFailsWith<IllegalArgumentException> {
            image.withColorSpace(ColorSpace.GRAYSCALE) // 1 channel vs 3
        }
    }

    @Test
    fun testWithColorSpaceRGBAFails() {
        val tensor = createTensor(Shape(3, 100, 100))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        assertFailsWith<IllegalArgumentException> {
            image.withColorSpace(ColorSpace.RGBA) // 4 channels vs 3
        }
    }

    // ========== Shape Validation Tests ==========

    @Test
    fun testTensorRankMismatchFails() {
        val tensor = createTensor(Shape(3, 100, 100)) // 3D tensor

        assertFailsWith<IllegalArgumentException> {
            Image.fromTensor(tensor, ImageLayout.NCHW, ColorSpace.RGB) // expects 4D
        }
    }

    @Test
    fun testTensorRank2Fails() {
        val tensor = createTensor(Shape(100, 100)) // 2D tensor

        assertFailsWith<IllegalArgumentException> {
            Image.fromTensor(tensor, ImageLayout.HWC, ColorSpace.GRAYSCALE) // expects 3D
        }
    }

    // ========== Shape Extension Tests ==========

    @Test
    fun testIsValidImageShapeHWC() {
        assertTrue(Shape(100, 200, 3).isValidImageShape(ImageLayout.HWC))
        assertFalse(Shape(100, 200).isValidImageShape(ImageLayout.HWC))
        assertFalse(Shape(1, 100, 200, 3).isValidImageShape(ImageLayout.HWC))
    }

    @Test
    fun testIsValidImageShapeNCHW() {
        assertTrue(Shape(8, 3, 100, 100).isValidImageShape(ImageLayout.NCHW))
        assertFalse(Shape(3, 100, 100).isValidImageShape(ImageLayout.NCHW))
    }

    @Test
    fun testImageDimensionsHWC() {
        val dims = Shape(480, 640, 3).imageDimensions(ImageLayout.HWC)

        assertEquals(640, dims?.width)
        assertEquals(480, dims?.height)
        assertEquals(3, dims?.channels)
        assertEquals(1, dims?.batchSize)
        assertEquals(480 * 640, dims?.pixelCount)
    }

    @Test
    fun testImageDimensionsNCHW() {
        val dims = Shape(32, 3, 224, 224).imageDimensions(ImageLayout.NCHW)

        assertEquals(224, dims?.width)
        assertEquals(224, dims?.height)
        assertEquals(3, dims?.channels)
        assertEquals(32, dims?.batchSize)
        assertEquals(224 * 224, dims?.pixelCount)
        assertEquals(32 * 3 * 224 * 224, dims?.totalElements)
    }

    @Test
    fun testImageDimensionsInvalidShape() {
        val dims = Shape(100, 100).imageDimensions(ImageLayout.HWC)
        assertEquals(null, dims)
    }

    // ========== toString Tests ==========

    @Test
    fun testToString() {
        val tensor = createTensor(Shape(3, 480, 640))
        val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)

        val str = image.toString()
        assertTrue(str.contains("640"))
        assertTrue(str.contains("480"))
        assertTrue(str.contains("3"))
        assertTrue(str.contains("CHW"))
        assertTrue(str.contains("RGB"))
    }
}
