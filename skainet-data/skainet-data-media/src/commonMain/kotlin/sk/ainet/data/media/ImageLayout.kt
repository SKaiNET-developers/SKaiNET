package sk.ainet.data.media

/**
 * Memory layout for image tensor data.
 *
 * Different frameworks use different conventions for storing image data:
 * - PyTorch typically uses CHW (channels first)
 * - TensorFlow typically uses HWC (channels last)
 * - Batched versions add a batch dimension at the front
 */
public enum class ImageLayout {
    /**
     * Height × Width × Channels (e.g., OpenCV, TensorFlow default).
     * Shape: [H, W, C]
     */
    HWC,

    /**
     * Channels × Height × Width (e.g., PyTorch default).
     * Shape: [C, H, W]
     */
    CHW,

    /**
     * Batch × Height × Width × Channels (e.g., TensorFlow batched).
     * Shape: [N, H, W, C]
     */
    NHWC,

    /**
     * Batch × Channels × Height × Width (e.g., PyTorch batched).
     * Shape: [N, C, H, W]
     */
    NCHW;

    /**
     * Whether this layout includes a batch dimension.
     */
    public val isBatched: Boolean
        get() = this == NHWC || this == NCHW

    /**
     * Whether channels are the first spatial dimension (after batch if present).
     */
    public val isChannelsFirst: Boolean
        get() = this == CHW || this == NCHW

    /**
     * Expected tensor rank for this layout.
     */
    public val expectedRank: Int
        get() = if (isBatched) 4 else 3

    /**
     * Index of the channel dimension in the shape array.
     */
    public val channelAxis: Int
        get() = when (this) {
            HWC -> 2
            CHW -> 0
            NHWC -> 3
            NCHW -> 1
        }

    /**
     * Index of the height dimension in the shape array.
     */
    public val heightAxis: Int
        get() = when (this) {
            HWC -> 0
            CHW -> 1
            NHWC -> 1
            NCHW -> 2
        }

    /**
     * Index of the width dimension in the shape array.
     */
    public val widthAxis: Int
        get() = when (this) {
            HWC -> 1
            CHW -> 2
            NHWC -> 2
            NCHW -> 3
        }

    /**
     * Convert to batched version of this layout.
     */
    public fun batched(): ImageLayout = when (this) {
        HWC -> NHWC
        CHW -> NCHW
        NHWC -> NHWC
        NCHW -> NCHW
    }

    /**
     * Convert to unbatched version of this layout.
     */
    public fun unbatched(): ImageLayout = when (this) {
        HWC -> HWC
        CHW -> CHW
        NHWC -> HWC
        NCHW -> CHW
    }
}
