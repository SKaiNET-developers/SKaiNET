package sk.ainet.data.media

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ImageLayoutTest {

    @Test
    fun testHWCProperties() {
        val layout = ImageLayout.HWC
        assertEquals(3, layout.expectedRank)
        assertFalse(layout.isBatched)
        assertFalse(layout.isChannelsFirst)
        assertEquals(0, layout.heightAxis)
        assertEquals(1, layout.widthAxis)
        assertEquals(2, layout.channelAxis)
    }

    @Test
    fun testCHWProperties() {
        val layout = ImageLayout.CHW
        assertEquals(3, layout.expectedRank)
        assertFalse(layout.isBatched)
        assertTrue(layout.isChannelsFirst)
        assertEquals(0, layout.channelAxis)
        assertEquals(1, layout.heightAxis)
        assertEquals(2, layout.widthAxis)
    }

    @Test
    fun testNHWCProperties() {
        val layout = ImageLayout.NHWC
        assertEquals(4, layout.expectedRank)
        assertTrue(layout.isBatched)
        assertFalse(layout.isChannelsFirst)
        assertEquals(1, layout.heightAxis)
        assertEquals(2, layout.widthAxis)
        assertEquals(3, layout.channelAxis)
    }

    @Test
    fun testNCHWProperties() {
        val layout = ImageLayout.NCHW
        assertEquals(4, layout.expectedRank)
        assertTrue(layout.isBatched)
        assertTrue(layout.isChannelsFirst)
        assertEquals(1, layout.channelAxis)
        assertEquals(2, layout.heightAxis)
        assertEquals(3, layout.widthAxis)
    }

    @Test
    fun testBatchedConversion() {
        assertEquals(ImageLayout.NHWC, ImageLayout.HWC.batched())
        assertEquals(ImageLayout.NCHW, ImageLayout.CHW.batched())
        assertEquals(ImageLayout.NHWC, ImageLayout.NHWC.batched())
        assertEquals(ImageLayout.NCHW, ImageLayout.NCHW.batched())
    }

    @Test
    fun testUnbatchedConversion() {
        assertEquals(ImageLayout.HWC, ImageLayout.HWC.unbatched())
        assertEquals(ImageLayout.CHW, ImageLayout.CHW.unbatched())
        assertEquals(ImageLayout.HWC, ImageLayout.NHWC.unbatched())
        assertEquals(ImageLayout.CHW, ImageLayout.NCHW.unbatched())
    }

    @Test
    fun testAxisConsistency() {
        // For CHW: shape is [C, H, W] so axes should be 0, 1, 2
        val chw = ImageLayout.CHW
        assertEquals(0, chw.channelAxis)
        assertEquals(1, chw.heightAxis)
        assertEquals(2, chw.widthAxis)

        // For NCHW: shape is [N, C, H, W] so axes should be 1, 2, 3
        val nchw = ImageLayout.NCHW
        assertEquals(1, nchw.channelAxis)
        assertEquals(2, nchw.heightAxis)
        assertEquals(3, nchw.widthAxis)
    }
}
