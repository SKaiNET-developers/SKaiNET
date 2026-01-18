package sk.ainet.data.media

/**
 * Memory layout for audio sample data.
 *
 * Different audio processing libraries use different conventions
 * for storing multi-channel audio data.
 */
public enum class ChannelLayout {
    /**
     * Single channel (mono) audio.
     * Shape: [samples]
     */
    MONO,

    /**
     * Interleaved multi-channel audio.
     * Samples from different channels alternate: L R L R L R...
     * Shape: [samples, channels]
     */
    INTERLEAVED,

    /**
     * Planar (non-interleaved) multi-channel audio.
     * Each channel is stored contiguously: LLLL...RRRR...
     * Shape: [channels, samples]
     */
    PLANAR,

    /**
     * Batched interleaved audio.
     * Shape: [batch, samples, channels]
     */
    BATCH_INTERLEAVED,

    /**
     * Batched planar audio.
     * Shape: [batch, channels, samples]
     */
    BATCH_PLANAR;

    /**
     * Whether this layout includes a batch dimension.
     */
    public val isBatched: Boolean
        get() = this == BATCH_INTERLEAVED || this == BATCH_PLANAR

    /**
     * Whether this layout is mono (single channel).
     */
    public val isMono: Boolean
        get() = this == MONO

    /**
     * Whether channels are stored contiguously (planar).
     */
    public val isPlanar: Boolean
        get() = this == PLANAR || this == BATCH_PLANAR

    /**
     * Expected tensor rank for this layout.
     */
    public val expectedRank: Int
        get() = when (this) {
            MONO -> 1
            INTERLEAVED, PLANAR -> 2
            BATCH_INTERLEAVED, BATCH_PLANAR -> 3
        }

    /**
     * Index of the samples dimension in the shape array.
     */
    public val samplesAxis: Int
        get() = when (this) {
            MONO -> 0
            INTERLEAVED -> 0
            PLANAR -> 1
            BATCH_INTERLEAVED -> 1
            BATCH_PLANAR -> 2
        }

    /**
     * Index of the channels dimension in the shape array.
     * Returns -1 for MONO (no channel dimension).
     */
    public val channelsAxis: Int
        get() = when (this) {
            MONO -> -1
            INTERLEAVED -> 1
            PLANAR -> 0
            BATCH_INTERLEAVED -> 2
            BATCH_PLANAR -> 1
        }

    /**
     * Convert to batched version of this layout.
     */
    public fun batched(): ChannelLayout = when (this) {
        MONO -> BATCH_PLANAR
        INTERLEAVED -> BATCH_INTERLEAVED
        PLANAR -> BATCH_PLANAR
        BATCH_INTERLEAVED -> BATCH_INTERLEAVED
        BATCH_PLANAR -> BATCH_PLANAR
    }

    /**
     * Convert to unbatched version of this layout.
     */
    public fun unbatched(): ChannelLayout = when (this) {
        MONO -> MONO
        INTERLEAVED -> INTERLEAVED
        PLANAR -> PLANAR
        BATCH_INTERLEAVED -> INTERLEAVED
        BATCH_PLANAR -> PLANAR
    }
}
