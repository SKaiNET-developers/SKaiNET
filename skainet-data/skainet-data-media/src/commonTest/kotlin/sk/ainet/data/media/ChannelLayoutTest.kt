package sk.ainet.data.media

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ChannelLayoutTest {

    @Test
    fun testMonoProperties() {
        val layout = ChannelLayout.MONO
        assertEquals(1, layout.expectedRank)
        assertFalse(layout.isBatched)
        assertTrue(layout.isMono)
        assertFalse(layout.isPlanar)
        assertEquals(0, layout.samplesAxis)
        assertEquals(-1, layout.channelsAxis)
    }

    @Test
    fun testInterleavedProperties() {
        val layout = ChannelLayout.INTERLEAVED
        assertEquals(2, layout.expectedRank)
        assertFalse(layout.isBatched)
        assertFalse(layout.isMono)
        assertFalse(layout.isPlanar)
        assertEquals(0, layout.samplesAxis)
        assertEquals(1, layout.channelsAxis)
    }

    @Test
    fun testPlanarProperties() {
        val layout = ChannelLayout.PLANAR
        assertEquals(2, layout.expectedRank)
        assertFalse(layout.isBatched)
        assertFalse(layout.isMono)
        assertTrue(layout.isPlanar)
        assertEquals(1, layout.samplesAxis)
        assertEquals(0, layout.channelsAxis)
    }

    @Test
    fun testBatchInterleavedProperties() {
        val layout = ChannelLayout.BATCH_INTERLEAVED
        assertEquals(3, layout.expectedRank)
        assertTrue(layout.isBatched)
        assertFalse(layout.isMono)
        assertFalse(layout.isPlanar)
        assertEquals(1, layout.samplesAxis)
        assertEquals(2, layout.channelsAxis)
    }

    @Test
    fun testBatchPlanarProperties() {
        val layout = ChannelLayout.BATCH_PLANAR
        assertEquals(3, layout.expectedRank)
        assertTrue(layout.isBatched)
        assertFalse(layout.isMono)
        assertTrue(layout.isPlanar)
        assertEquals(2, layout.samplesAxis)
        assertEquals(1, layout.channelsAxis)
    }

    @Test
    fun testBatchedConversion() {
        assertEquals(ChannelLayout.BATCH_PLANAR, ChannelLayout.MONO.batched())
        assertEquals(ChannelLayout.BATCH_INTERLEAVED, ChannelLayout.INTERLEAVED.batched())
        assertEquals(ChannelLayout.BATCH_PLANAR, ChannelLayout.PLANAR.batched())
        assertEquals(ChannelLayout.BATCH_INTERLEAVED, ChannelLayout.BATCH_INTERLEAVED.batched())
        assertEquals(ChannelLayout.BATCH_PLANAR, ChannelLayout.BATCH_PLANAR.batched())
    }

    @Test
    fun testUnbatchedConversion() {
        assertEquals(ChannelLayout.MONO, ChannelLayout.MONO.unbatched())
        assertEquals(ChannelLayout.INTERLEAVED, ChannelLayout.INTERLEAVED.unbatched())
        assertEquals(ChannelLayout.PLANAR, ChannelLayout.PLANAR.unbatched())
        assertEquals(ChannelLayout.INTERLEAVED, ChannelLayout.BATCH_INTERLEAVED.unbatched())
        assertEquals(ChannelLayout.PLANAR, ChannelLayout.BATCH_PLANAR.unbatched())
    }

    @Test
    fun testAllLayoutsHaveCorrectRank() {
        // Verify rank increases with batching
        for (layout in listOf(ChannelLayout.MONO)) {
            assertEquals(1, layout.expectedRank)
        }
        for (layout in listOf(ChannelLayout.INTERLEAVED, ChannelLayout.PLANAR)) {
            assertEquals(2, layout.expectedRank)
        }
        for (layout in listOf(ChannelLayout.BATCH_INTERLEAVED, ChannelLayout.BATCH_PLANAR)) {
            assertEquals(3, layout.expectedRank)
        }
    }

    @Test
    fun testPlanarLayouts() {
        val planarLayouts = listOf(ChannelLayout.PLANAR, ChannelLayout.BATCH_PLANAR)
        for (layout in planarLayouts) {
            assertTrue(layout.isPlanar, "${layout.name} should be planar")
        }

        val nonPlanarLayouts = listOf(ChannelLayout.MONO, ChannelLayout.INTERLEAVED, ChannelLayout.BATCH_INTERLEAVED)
        for (layout in nonPlanarLayouts) {
            assertFalse(layout.isPlanar, "${layout.name} should not be planar")
        }
    }

    @Test
    fun testBatchedLayouts() {
        val batchedLayouts = listOf(ChannelLayout.BATCH_INTERLEAVED, ChannelLayout.BATCH_PLANAR)
        for (layout in batchedLayouts) {
            assertTrue(layout.isBatched, "${layout.name} should be batched")
        }

        val unbatchedLayouts = listOf(ChannelLayout.MONO, ChannelLayout.INTERLEAVED, ChannelLayout.PLANAR)
        for (layout in unbatchedLayouts) {
            assertFalse(layout.isBatched, "${layout.name} should not be batched")
        }
    }

    @Test
    fun testAxisConsistencyInterleaved() {
        // For INTERLEAVED: shape is [samples, channels] so axes should be 0, 1
        val interleaved = ChannelLayout.INTERLEAVED
        assertEquals(0, interleaved.samplesAxis)
        assertEquals(1, interleaved.channelsAxis)
    }

    @Test
    fun testAxisConsistencyPlanar() {
        // For PLANAR: shape is [channels, samples] so axes should be 1, 0
        val planar = ChannelLayout.PLANAR
        assertEquals(1, planar.samplesAxis)
        assertEquals(0, planar.channelsAxis)
    }

    @Test
    fun testAxisConsistencyBatchInterleaved() {
        // For BATCH_INTERLEAVED: shape is [batch, samples, channels] so axes should be 1, 2
        val layout = ChannelLayout.BATCH_INTERLEAVED
        assertEquals(1, layout.samplesAxis)
        assertEquals(2, layout.channelsAxis)
    }

    @Test
    fun testAxisConsistencyBatchPlanar() {
        // For BATCH_PLANAR: shape is [batch, channels, samples] so axes should be 2, 1
        val layout = ChannelLayout.BATCH_PLANAR
        assertEquals(2, layout.samplesAxis)
        assertEquals(1, layout.channelsAxis)
    }

    @Test
    fun testRoundTripBatchConversion() {
        // unbatched -> batched -> unbatched should return original (except MONO)
        assertEquals(ChannelLayout.INTERLEAVED, ChannelLayout.INTERLEAVED.batched().unbatched())
        assertEquals(ChannelLayout.PLANAR, ChannelLayout.PLANAR.batched().unbatched())
    }
}
