package sk.ainet.data.media

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * Image wrapper that combines tensor data with image-specific metadata.
 *
 * This class provides a type-safe representation of image data for use in
 * data processing pipelines. It wraps an underlying tensor and tracks
 * layout and color space information.
 *
 * Example:
 * ```kotlin
 * val image = Image.fromTensor(tensor, ImageLayout.CHW, ColorSpace.RGB)
 * println("Size: ${image.width} x ${image.height}")
 * println("Channels: ${image.channels}")
 * ```
 *
 * @param T The DType of the underlying tensor
 * @param V The value type of tensor elements
 */
public class Image<T : DType, V> private constructor(
    /**
     * The underlying tensor data.
     */
    public val tensor: Tensor<T, V>,

    /**
     * Memory layout of the image data.
     */
    public val layout: ImageLayout,

    /**
     * Color space interpretation.
     */
    public val colorSpace: ColorSpace
) {
    /**
     * Image width in pixels.
     */
    public val width: Int
        get() = tensor.shape[layout.widthAxis]

    /**
     * Image height in pixels.
     */
    public val height: Int
        get() = tensor.shape[layout.heightAxis]

    /**
     * Number of color channels.
     */
    public val channels: Int
        get() = tensor.shape[layout.channelAxis]

    /**
     * Batch size (1 if not batched).
     */
    public val batchSize: Int
        get() = if (layout.isBatched) tensor.shape[0] else 1

    /**
     * Whether this image has a batch dimension.
     */
    public val isBatched: Boolean
        get() = layout.isBatched

    /**
     * Total number of pixels (height × width).
     */
    public val pixelCount: Int
        get() = width * height

    /**
     * The shape of the underlying tensor.
     */
    public val shape: Shape
        get() = tensor.shape

    /**
     * Check if channel count matches color space.
     */
    public val isConsistent: Boolean
        get() = channels == colorSpace.channels

    /**
     * Create a copy with different layout (metadata only, no data transformation).
     *
     * Note: This only changes the metadata. Use transpose for actual data transformation.
     */
    public fun withLayout(newLayout: ImageLayout): Image<T, V> {
        require(newLayout.expectedRank == layout.expectedRank) {
            "Cannot change layout from ${layout.expectedRank}D to ${newLayout.expectedRank}D without reshaping"
        }
        return Image(tensor, newLayout, colorSpace)
    }

    /**
     * Create a copy with different color space interpretation (metadata only).
     *
     * Note: This only changes the metadata. Use conversion functions for actual color transformation.
     */
    public fun withColorSpace(newColorSpace: ColorSpace): Image<T, V> {
        require(newColorSpace.channels == colorSpace.channels) {
            "Cannot change color space from ${colorSpace.channels} to ${newColorSpace.channels} channels without conversion"
        }
        return Image(tensor, layout, newColorSpace)
    }

    override fun toString(): String {
        return "Image(${width}×${height}, channels=$channels, layout=$layout, colorSpace=$colorSpace)"
    }

    public companion object {
        /**
         * Create an Image from an existing tensor with explicit metadata.
         *
         * @param tensor The tensor data
         * @param layout Memory layout of the tensor
         * @param colorSpace Color space interpretation
         * @return A new Image wrapping the tensor
         * @throws IllegalArgumentException if tensor rank doesn't match layout
         */
        public fun <T : DType, V> fromTensor(
            tensor: Tensor<T, V>,
            layout: ImageLayout,
            colorSpace: ColorSpace
        ): Image<T, V> {
            require(tensor.rank == layout.expectedRank) {
                "Tensor rank ${tensor.rank} doesn't match expected rank ${layout.expectedRank} for layout $layout"
            }
            return Image(tensor, layout, colorSpace)
        }

        /**
         * Create an Image from a tensor, inferring color space from channel count.
         *
         * @param tensor The tensor data
         * @param layout Memory layout of the tensor
         * @return A new Image with inferred color space
         */
        public fun <T : DType, V> fromTensor(
            tensor: Tensor<T, V>,
            layout: ImageLayout
        ): Image<T, V> {
            require(tensor.rank == layout.expectedRank) {
                "Tensor rank ${tensor.rank} doesn't match expected rank ${layout.expectedRank} for layout $layout"
            }
            val channels = tensor.shape[layout.channelAxis]
            val colorSpace = inferColorSpace(channels)
            return Image(tensor, layout, colorSpace)
        }

        /**
         * Infer color space from channel count.
         */
        private fun inferColorSpace(channels: Int): ColorSpace = when (channels) {
            1 -> ColorSpace.GRAYSCALE
            3 -> ColorSpace.RGB
            4 -> ColorSpace.RGBA
            else -> throw IllegalArgumentException(
                "Cannot infer color space for $channels channels. Use explicit colorSpace parameter."
            )
        }
    }
}

/**
 * Extension to check if a tensor shape is compatible with an image layout.
 */
public fun Shape.isValidImageShape(layout: ImageLayout): Boolean {
    return rank == layout.expectedRank
}

/**
 * Extension to extract image dimensions from a shape given a layout.
 */
public fun Shape.imageDimensions(layout: ImageLayout): ImageDimensions? {
    if (!isValidImageShape(layout)) return null
    return ImageDimensions(
        width = this[layout.widthAxis],
        height = this[layout.heightAxis],
        channels = this[layout.channelAxis],
        batchSize = if (layout.isBatched) this[0] else 1
    )
}

/**
 * Image dimension information.
 */
public data class ImageDimensions(
    public val width: Int,
    public val height: Int,
    public val channels: Int,
    public val batchSize: Int = 1
) {
    public val pixelCount: Int get() = width * height
    public val totalElements: Int get() = batchSize * channels * height * width
}
