package sk.ainet.data.media

/**
 * Color space for image data.
 *
 * Defines how color values are interpreted in an image tensor.
 */
public enum class ColorSpace(
    /**
     * Number of channels for this color space.
     */
    public val channels: Int
) {
    /**
     * Single-channel grayscale.
     */
    GRAYSCALE(1),

    /**
     * Red, Green, Blue (standard RGB order).
     */
    RGB(3),

    /**
     * Blue, Green, Red (OpenCV default order).
     */
    BGR(3),

    /**
     * Red, Green, Blue, Alpha (with transparency).
     */
    RGBA(4),

    /**
     * Blue, Green, Red, Alpha.
     */
    BGRA(4),

    /**
     * YUV color space (luminance + chrominance).
     */
    YUV(3),

    /**
     * Hue, Saturation, Value.
     */
    HSV(3),

    /**
     * CIE LAB color space.
     */
    LAB(3);

    /**
     * Whether this color space has an alpha (transparency) channel.
     */
    public val hasAlpha: Boolean
        get() = this == RGBA || this == BGRA

    /**
     * Whether this is a grayscale (single-channel) color space.
     */
    public val isGrayscale: Boolean
        get() = this == GRAYSCALE
}
