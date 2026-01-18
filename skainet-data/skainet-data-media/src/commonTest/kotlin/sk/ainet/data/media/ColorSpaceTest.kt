package sk.ainet.data.media

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ColorSpaceTest {

    @Test
    fun testChannelCounts() {
        assertEquals(1, ColorSpace.GRAYSCALE.channels)
        assertEquals(3, ColorSpace.RGB.channels)
        assertEquals(3, ColorSpace.BGR.channels)
        assertEquals(4, ColorSpace.RGBA.channels)
        assertEquals(4, ColorSpace.BGRA.channels)
        assertEquals(3, ColorSpace.YUV.channels)
        assertEquals(3, ColorSpace.HSV.channels)
        assertEquals(3, ColorSpace.LAB.channels)
    }

    @Test
    fun testGrayscaleFlag() {
        assertTrue(ColorSpace.GRAYSCALE.isGrayscale)
        assertFalse(ColorSpace.RGB.isGrayscale)
        assertFalse(ColorSpace.BGR.isGrayscale)
        assertFalse(ColorSpace.RGBA.isGrayscale)
        assertFalse(ColorSpace.YUV.isGrayscale)
    }

    @Test
    fun testAlphaFlag() {
        assertFalse(ColorSpace.GRAYSCALE.hasAlpha)
        assertFalse(ColorSpace.RGB.hasAlpha)
        assertFalse(ColorSpace.BGR.hasAlpha)
        assertTrue(ColorSpace.RGBA.hasAlpha)
        assertTrue(ColorSpace.BGRA.hasAlpha)
        assertFalse(ColorSpace.YUV.hasAlpha)
        assertFalse(ColorSpace.HSV.hasAlpha)
        assertFalse(ColorSpace.LAB.hasAlpha)
    }

    @Test
    fun testThreeChannelSpaces() {
        val threeChannelSpaces = listOf(
            ColorSpace.RGB,
            ColorSpace.BGR,
            ColorSpace.YUV,
            ColorSpace.HSV,
            ColorSpace.LAB
        )
        for (space in threeChannelSpaces) {
            assertEquals(3, space.channels, "${space.name} should have 3 channels")
            assertFalse(space.isGrayscale, "${space.name} should not be grayscale")
        }
    }

    @Test
    fun testFourChannelSpaces() {
        val fourChannelSpaces = listOf(ColorSpace.RGBA, ColorSpace.BGRA)
        for (space in fourChannelSpaces) {
            assertEquals(4, space.channels, "${space.name} should have 4 channels")
            assertTrue(space.hasAlpha, "${space.name} should have alpha")
        }
    }
}
