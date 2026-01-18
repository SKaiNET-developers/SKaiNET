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

class AudioTest {

    private val dataFactory = DenseTensorDataFactory()

    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    // ========== MONO Layout Tests ==========

    @Test
    fun testMonoAudio() {
        val tensor = createTensor(Shape(16000)) // 1 second at 16kHz
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)

        assertEquals(16000, audio.sampleCount)
        assertEquals(1, audio.channelCount)
        assertEquals(1.0, audio.duration, 0.0001)
        assertEquals(1000.0, audio.durationMs, 0.1)
        assertEquals(1, audio.batchSize)
        assertFalse(audio.isBatched)
        assertTrue(audio.isMono)
        assertFalse(audio.isStereo)
    }

    @Test
    fun testMonoAudioHalfSecond() {
        val tensor = createTensor(Shape(22050)) // 0.5 second at 44.1kHz
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_44K, ChannelLayout.MONO)

        assertEquals(22050, audio.sampleCount)
        assertEquals(0.5, audio.duration, 0.0001)
        assertEquals(500.0, audio.durationMs, 0.1)
    }

    // ========== INTERLEAVED Layout Tests ==========

    @Test
    fun testInterleavedStereo() {
        val tensor = createTensor(Shape(48000, 2)) // 1 second at 48kHz, stereo
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_48K, ChannelLayout.INTERLEAVED)

        assertEquals(48000, audio.sampleCount)
        assertEquals(2, audio.channelCount)
        assertEquals(1.0, audio.duration, 0.0001)
        assertFalse(audio.isBatched)
        assertFalse(audio.isMono)
        assertTrue(audio.isStereo)
    }

    @Test
    fun testInterleavedMultichannel() {
        val tensor = createTensor(Shape(16000, 6)) // 5.1 surround
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_16K, ChannelLayout.INTERLEAVED)

        assertEquals(16000, audio.sampleCount)
        assertEquals(6, audio.channelCount)
        assertFalse(audio.isMono)
        assertFalse(audio.isStereo)
    }

    // ========== PLANAR Layout Tests ==========

    @Test
    fun testPlanarStereo() {
        val tensor = createTensor(Shape(2, 44100)) // stereo at 44.1kHz
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_44K, ChannelLayout.PLANAR)

        assertEquals(44100, audio.sampleCount)
        assertEquals(2, audio.channelCount)
        assertEquals(1.0, audio.duration, 0.0001)
        assertFalse(audio.isBatched)
        assertTrue(audio.isStereo)
    }

    @Test
    fun testPlanarMono() {
        val tensor = createTensor(Shape(1, 8000))
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_8K, ChannelLayout.PLANAR)

        assertEquals(8000, audio.sampleCount)
        assertEquals(1, audio.channelCount)
        assertTrue(audio.isMono)
    }

    // ========== BATCH_INTERLEAVED Layout Tests ==========

    @Test
    fun testBatchInterleavedStereo() {
        val tensor = createTensor(Shape(32, 16000, 2)) // batch of 32, 1 second, stereo
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_16K, ChannelLayout.BATCH_INTERLEAVED)

        assertEquals(16000, audio.sampleCount)
        assertEquals(2, audio.channelCount)
        assertEquals(32, audio.batchSize)
        assertTrue(audio.isBatched)
        assertTrue(audio.isStereo)
    }

    @Test
    fun testBatchInterleavedMono() {
        val tensor = createTensor(Shape(16, 8000, 1)) // batch of 16, mono
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_8K, ChannelLayout.BATCH_INTERLEAVED)

        assertEquals(8000, audio.sampleCount)
        assertEquals(1, audio.channelCount)
        assertEquals(16, audio.batchSize)
        assertTrue(audio.isBatched)
        assertTrue(audio.isMono)
    }

    // ========== BATCH_PLANAR Layout Tests ==========

    @Test
    fun testBatchPlanarStereo() {
        val tensor = createTensor(Shape(8, 2, 22050)) // batch of 8, stereo, 0.5 second at 44.1kHz
        val audio = Audio.fromTensor(tensor, Audio.SAMPLE_RATE_44K, ChannelLayout.BATCH_PLANAR)

        assertEquals(22050, audio.sampleCount)
        assertEquals(2, audio.channelCount)
        assertEquals(8, audio.batchSize)
        assertEquals(0.5, audio.duration, 0.0001)
        assertTrue(audio.isBatched)
        assertTrue(audio.isStereo)
    }

    // ========== Sample Rate Constants Tests ==========

    @Test
    fun testSampleRateConstants() {
        assertEquals(8000, Audio.SAMPLE_RATE_8K)
        assertEquals(16000, Audio.SAMPLE_RATE_16K)
        assertEquals(22050, Audio.SAMPLE_RATE_22K)
        assertEquals(44100, Audio.SAMPLE_RATE_44K)
        assertEquals(48000, Audio.SAMPLE_RATE_48K)
    }

    // ========== Duration Calculation Tests ==========

    @Test
    fun testDurationCalculation() {
        // 16000 samples at 16000 Hz = 1 second
        val tensor1 = createTensor(Shape(16000))
        val audio1 = Audio.fromTensor(tensor1, 16000, ChannelLayout.MONO)
        assertEquals(1.0, audio1.duration, 0.0001)

        // 44100 samples at 44100 Hz = 1 second
        val tensor2 = createTensor(Shape(44100))
        val audio2 = Audio.fromTensor(tensor2, 44100, ChannelLayout.MONO)
        assertEquals(1.0, audio2.duration, 0.0001)

        // 24000 samples at 48000 Hz = 0.5 seconds
        val tensor3 = createTensor(Shape(24000))
        val audio3 = Audio.fromTensor(tensor3, 48000, ChannelLayout.MONO)
        assertEquals(0.5, audio3.duration, 0.0001)
    }

    // ========== withSampleRate Tests ==========

    @Test
    fun testWithSampleRate() {
        val tensor = createTensor(Shape(16000))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)

        val resampled = audio.withSampleRate(8000)

        assertEquals(8000, resampled.sampleRate)
        assertEquals(16000, resampled.sampleCount) // Same samples
        assertEquals(2.0, resampled.duration, 0.0001) // But different duration
    }

    @Test
    fun testWithSampleRateZeroFails() {
        val tensor = createTensor(Shape(16000))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)

        assertFailsWith<IllegalArgumentException> {
            audio.withSampleRate(0)
        }
    }

    @Test
    fun testWithSampleRateNegativeFails() {
        val tensor = createTensor(Shape(16000))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)

        assertFailsWith<IllegalArgumentException> {
            audio.withSampleRate(-1)
        }
    }

    // ========== withLayout Tests ==========

    @Test
    fun testWithLayoutSameRank() {
        val tensor = createTensor(Shape(16000, 2))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.INTERLEAVED)

        val reinterpreted = audio.withLayout(ChannelLayout.PLANAR)

        assertEquals(ChannelLayout.PLANAR, reinterpreted.layout)
        assertEquals(16000, reinterpreted.sampleRate)
    }

    @Test
    fun testWithLayoutDifferentRankFails() {
        val tensor = createTensor(Shape(16000, 2))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.INTERLEAVED)

        assertFailsWith<IllegalArgumentException> {
            audio.withLayout(ChannelLayout.BATCH_INTERLEAVED) // 3D vs 2D
        }
    }

    @Test
    fun testWithLayoutBatchedToBatched() {
        val tensor = createTensor(Shape(8, 16000, 2))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.BATCH_INTERLEAVED)

        val reinterpreted = audio.withLayout(ChannelLayout.BATCH_PLANAR)

        assertEquals(ChannelLayout.BATCH_PLANAR, reinterpreted.layout)
    }

    // ========== Validation Tests ==========

    @Test
    fun testInvalidSampleRateZero() {
        val tensor = createTensor(Shape(16000))

        assertFailsWith<IllegalArgumentException> {
            Audio.fromTensor(tensor, 0, ChannelLayout.MONO)
        }
    }

    @Test
    fun testInvalidSampleRateNegative() {
        val tensor = createTensor(Shape(16000))

        assertFailsWith<IllegalArgumentException> {
            Audio.fromTensor(tensor, -16000, ChannelLayout.MONO)
        }
    }

    @Test
    fun testTensorRankMismatch() {
        val tensor = createTensor(Shape(16000)) // 1D tensor

        assertFailsWith<IllegalArgumentException> {
            Audio.fromTensor(tensor, 16000, ChannelLayout.INTERLEAVED) // expects 2D
        }
    }

    @Test
    fun testTensorRankMismatchBatched() {
        val tensor = createTensor(Shape(16000, 2)) // 2D tensor

        assertFailsWith<IllegalArgumentException> {
            Audio.fromTensor(tensor, 16000, ChannelLayout.BATCH_INTERLEAVED) // expects 3D
        }
    }

    // ========== Shape Extension Tests ==========

    @Test
    fun testIsValidAudioShapeMono() {
        assertTrue(Shape(16000).isValidAudioShape(ChannelLayout.MONO))
        assertFalse(Shape(16000, 2).isValidAudioShape(ChannelLayout.MONO))
    }

    @Test
    fun testIsValidAudioShapeInterleaved() {
        assertTrue(Shape(16000, 2).isValidAudioShape(ChannelLayout.INTERLEAVED))
        assertFalse(Shape(16000).isValidAudioShape(ChannelLayout.INTERLEAVED))
    }

    @Test
    fun testIsValidAudioShapeBatchPlanar() {
        assertTrue(Shape(8, 2, 16000).isValidAudioShape(ChannelLayout.BATCH_PLANAR))
        assertFalse(Shape(2, 16000).isValidAudioShape(ChannelLayout.BATCH_PLANAR))
    }

    @Test
    fun testAudioDimensionsMono() {
        val dims = Shape(16000).audioDimensions(ChannelLayout.MONO)

        assertEquals(16000, dims?.sampleCount)
        assertEquals(1, dims?.channelCount)
        assertEquals(1, dims?.batchSize)
        assertEquals(16000, dims?.totalSamples)
    }

    @Test
    fun testAudioDimensionsInterleaved() {
        val dims = Shape(44100, 2).audioDimensions(ChannelLayout.INTERLEAVED)

        assertEquals(44100, dims?.sampleCount)
        assertEquals(2, dims?.channelCount)
        assertEquals(1, dims?.batchSize)
        assertEquals(44100 * 2, dims?.totalSamples)
    }

    @Test
    fun testAudioDimensionsBatchPlanar() {
        val dims = Shape(16, 2, 8000).audioDimensions(ChannelLayout.BATCH_PLANAR)

        assertEquals(8000, dims?.sampleCount)
        assertEquals(2, dims?.channelCount)
        assertEquals(16, dims?.batchSize)
        assertEquals(16 * 2 * 8000, dims?.totalSamples)
    }

    @Test
    fun testAudioDimensionsInvalidShape() {
        val dims = Shape(16000).audioDimensions(ChannelLayout.INTERLEAVED)
        assertEquals(null, dims)
    }

    @Test
    fun testAudioDimensionsDuration() {
        val dims = Shape(16000).audioDimensions(ChannelLayout.MONO)!!
        assertEquals(1.0, dims.duration(16000), 0.0001)
        assertEquals(0.5, dims.duration(32000), 0.0001)
    }

    // ========== toString Tests ==========

    @Test
    fun testToString() {
        val tensor = createTensor(Shape(16000))
        val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)

        val str = audio.toString()
        assertTrue(str.contains("16000"))
        assertTrue(str.contains("MONO"))
    }

    @Test
    fun testToStringStereo() {
        val tensor = createTensor(Shape(44100, 2))
        val audio = Audio.fromTensor(tensor, 44100, ChannelLayout.INTERLEAVED)

        val str = audio.toString()
        assertTrue(str.contains("44100"))
        assertTrue(str.contains("INTERLEAVED"))
        assertTrue(str.contains("2"))
    }
}
