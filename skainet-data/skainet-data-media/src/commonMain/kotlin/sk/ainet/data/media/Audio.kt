package sk.ainet.data.media

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * Audio wrapper that combines tensor data with audio-specific metadata.
 *
 * This class provides a type-safe representation of audio data for use in
 * data processing pipelines. It wraps an underlying tensor and tracks
 * sample rate, channel layout, and other audio properties.
 *
 * Example:
 * ```kotlin
 * val audio = Audio.fromTensor(tensor, 16000, ChannelLayout.MONO)
 * println("Duration: ${audio.duration} seconds")
 * println("Samples: ${audio.sampleCount}")
 * ```
 *
 * @param T The DType of the underlying tensor
 * @param V The value type of tensor elements
 */
public class Audio<T : DType, V> private constructor(
    /**
     * The underlying tensor data containing audio samples.
     */
    public val tensor: Tensor<T, V>,

    /**
     * Sample rate in Hz (samples per second).
     */
    public val sampleRate: Int,

    /**
     * Memory layout of the audio data.
     */
    public val layout: ChannelLayout
) {
    /**
     * Number of audio samples (per channel).
     */
    public val sampleCount: Int
        get() = tensor.shape[layout.samplesAxis]

    /**
     * Number of audio channels (1 for mono, 2 for stereo, etc.).
     */
    public val channelCount: Int
        get() = if (layout.isMono) 1 else tensor.shape[layout.channelsAxis]

    /**
     * Duration of the audio in seconds.
     */
    public val duration: Double
        get() = sampleCount.toDouble() / sampleRate

    /**
     * Duration in milliseconds.
     */
    public val durationMs: Double
        get() = duration * 1000.0

    /**
     * Batch size (1 if not batched).
     */
    public val batchSize: Int
        get() = if (layout.isBatched) tensor.shape[0] else 1

    /**
     * Whether this audio has a batch dimension.
     */
    public val isBatched: Boolean
        get() = layout.isBatched

    /**
     * Whether this is mono (single channel) audio.
     */
    public val isMono: Boolean
        get() = channelCount == 1

    /**
     * Whether this is stereo (two channel) audio.
     */
    public val isStereo: Boolean
        get() = channelCount == 2

    /**
     * The shape of the underlying tensor.
     */
    public val shape: Shape
        get() = tensor.shape

    /**
     * Create a copy with different sample rate (metadata only).
     *
     * Note: This only changes the metadata. Use resampling for actual rate conversion.
     */
    public fun withSampleRate(newSampleRate: Int): Audio<T, V> {
        require(newSampleRate > 0) { "Sample rate must be positive, got $newSampleRate" }
        return Audio(tensor, newSampleRate, layout)
    }

    /**
     * Create a copy with different layout (metadata only).
     *
     * Note: This only changes the metadata. Use layout conversion for actual data transformation.
     */
    public fun withLayout(newLayout: ChannelLayout): Audio<T, V> {
        require(newLayout.expectedRank == layout.expectedRank) {
            "Cannot change layout from ${layout.expectedRank}D to ${newLayout.expectedRank}D without reshaping"
        }
        return Audio(tensor, sampleRate, newLayout)
    }

    override fun toString(): String {
        return "Audio(${duration.format(2)}s @ ${sampleRate}Hz, channels=$channelCount, layout=$layout)"
    }

    public companion object {
        /**
         * Create Audio from an existing tensor with explicit metadata.
         *
         * @param tensor The tensor data containing audio samples
         * @param sampleRate Sample rate in Hz
         * @param layout Memory layout of the tensor
         * @return A new Audio wrapping the tensor
         * @throws IllegalArgumentException if tensor rank doesn't match layout
         */
        public fun <T : DType, V> fromTensor(
            tensor: Tensor<T, V>,
            sampleRate: Int,
            layout: ChannelLayout
        ): Audio<T, V> {
            require(sampleRate > 0) { "Sample rate must be positive, got $sampleRate" }
            require(tensor.rank == layout.expectedRank) {
                "Tensor rank ${tensor.rank} doesn't match expected rank ${layout.expectedRank} for layout $layout"
            }
            return Audio(tensor, sampleRate, layout)
        }

        /**
         * Common sample rates.
         */
        public const val SAMPLE_RATE_8K: Int = 8000
        public const val SAMPLE_RATE_16K: Int = 16000
        public const val SAMPLE_RATE_22K: Int = 22050
        public const val SAMPLE_RATE_44K: Int = 44100
        public const val SAMPLE_RATE_48K: Int = 48000
    }
}

/**
 * Extension to check if a tensor shape is compatible with an audio layout.
 */
public fun Shape.isValidAudioShape(layout: ChannelLayout): Boolean {
    return rank == layout.expectedRank
}

/**
 * Extension to extract audio dimensions from a shape given a layout.
 */
public fun Shape.audioDimensions(layout: ChannelLayout): AudioDimensions? {
    if (!isValidAudioShape(layout)) return null
    return AudioDimensions(
        sampleCount = this[layout.samplesAxis],
        channelCount = if (layout.isMono) 1 else this[layout.channelsAxis],
        batchSize = if (layout.isBatched) this[0] else 1
    )
}

/**
 * Audio dimension information.
 */
public data class AudioDimensions(
    public val sampleCount: Int,
    public val channelCount: Int,
    public val batchSize: Int = 1
) {
    public val totalSamples: Int get() = batchSize * channelCount * sampleCount

    public fun duration(sampleRate: Int): Double = sampleCount.toDouble() / sampleRate
}

/**
 * Format a double with specified decimal places.
 */
private fun Double.format(decimals: Int): String {
    var result = this
    repeat(decimals) { result *= 10 }
    val rounded = kotlin.math.round(result)
    repeat(decimals) { result = rounded / 10 }
    return result.toString()
}
