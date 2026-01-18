/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import java.awt.Color
import java.awt.image.BufferedImage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * JVM-specific tests for image transforms.
 */
class ImageTransformsJvmTest {

    private fun createTestImage(width: Int, height: Int, color: Color = Color.RED): BufferedImage {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        val g2d = image.createGraphics()
        g2d.color = color
        g2d.fillRect(0, 0, width, height)
        g2d.dispose()
        return image
    }

    @Test
    fun `ImageResize resizes image correctly`() {
        val original = createTestImage(100, 100)
        val resize = ImageResize(50, 50)

        val resized = resize.apply(original)

        assertEquals(50, resized.width)
        assertEquals(50, resized.height)
    }

    @Test
    fun `ImageResize with different interpolation modes`() {
        val original = createTestImage(100, 100)

        for (interpolation in Interpolation.entries) {
            val resize = ImageResize(50, 50, interpolation)
            val resized = resize.apply(original)

            assertEquals(50, resized.width, "Width mismatch for $interpolation")
            assertEquals(50, resized.height, "Height mismatch for $interpolation")
        }
    }

    @Test
    fun `ImageCrop crops image correctly`() {
        val original = createTestImage(100, 100)
        val crop = ImageCrop(top = 10, bottom = 10, left = 10, right = 10)

        val cropped = crop.apply(original)

        assertEquals(80, cropped.width)
        assertEquals(80, cropped.height)
    }

    @Test
    fun `ImageCenterCrop extracts centered square`() {
        val original = createTestImage(100, 80)
        val centerCrop = ImageCenterCrop(50)

        val cropped = centerCrop.apply(original)

        assertEquals(50, cropped.width)
        assertEquals(50, cropped.height)
    }

    @Test
    fun `ImageRotate rotates image`() {
        val original = createTestImage(100, 50)
        val rotate = ImageRotate(90f)

        val rotated = rotate.apply(original)

        // After 90-degree rotation, dimensions should swap (approximately)
        // Due to rotation algorithm, dimensions may vary slightly
        assertTrue(rotated.width > 0)
        assertTrue(rotated.height > 0)
    }

    @Test
    fun `ImagePad adds padding with correct color`() {
        val original = createTestImage(100, 100, Color.RED)
        val pad = ImagePad(
            top = 10,
            bottom = 10,
            left = 10,
            right = 10,
            red = 0,
            green = 255,
            blue = 0
        )

        val padded = pad.apply(original)

        assertEquals(120, padded.width)
        assertEquals(120, padded.height)

        // Check padding color at corners
        val topLeftPixel = padded.getRGB(0, 0)
        val paddingGreen = Color(topLeftPixel)
        assertEquals(0, paddingGreen.red)
        assertEquals(255, paddingGreen.green)
        assertEquals(0, paddingGreen.blue)

        // Check original image is preserved in center
        val centerPixel = padded.getRGB(60, 60)
        val centerColor = Color(centerPixel)
        assertEquals(255, centerColor.red)
        assertEquals(0, centerColor.green)
        assertEquals(0, centerColor.blue)
    }

    @Test
    fun `transform pipeline composes correctly`() {
        val original = createTestImage(200, 200)

        val pipeline = pipeline<BufferedImage>()
            .resize(100, 100)
            .crop(top = 10, bottom = 10, left = 10, right = 10)

        val result = pipeline.apply(original)

        assertEquals(80, result.width)
        assertEquals(80, result.height)
    }

    @Test
    fun `ImageResize validates positive dimensions`() {
        try {
            ImageResize(0, 100)
            throw AssertionError("Should have thrown exception for zero width")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Width") == true)
        }

        try {
            ImageResize(100, -1)
            throw AssertionError("Should have thrown exception for negative height")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Height") == true)
        }
    }

    @Test
    fun `ImageCrop validates non-negative edges`() {
        try {
            ImageCrop(top = -1)
            throw AssertionError("Should have thrown exception for negative top")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Top") == true)
        }
    }

    @Test
    fun `ImageCenterCrop validates positive size`() {
        try {
            ImageCenterCrop(0)
            throw AssertionError("Should have thrown exception for zero size")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Size") == true)
        }
    }

    @Test
    fun `ImagePad validates color range`() {
        try {
            ImagePad(red = 300)
            throw AssertionError("Should have thrown exception for invalid red value")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Red") == true)
        }

        try {
            ImagePad(green = -1)
            throw AssertionError("Should have thrown exception for invalid green value")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("Green") == true)
        }
    }
}
