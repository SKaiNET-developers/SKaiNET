/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.context.ExecutionContext
import sk.ainet.io.image.PlatformBitmapImage
import sk.ainet.io.image.platformImageToArgb
import sk.ainet.io.image.platformImageSize
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

/**
 * Interpolation methods for image resizing operations.
 */
public enum class Interpolation {
    /** Nearest neighbor - fastest, lowest quality */
    NEAREST,
    /** Bilinear interpolation - good balance of speed and quality */
    BILINEAR,
    /** Bicubic interpolation - highest quality, slower */
    BICUBIC
}

/**
 * Color modes for image processing.
 */
public enum class ColorMode {
    /** Red, Green, Blue (3 channels) */
    RGB,
    /** Blue, Green, Red (3 channels) - OpenCV default */
    BGR,
    /** Single channel grayscale */
    GRAYSCALE
}

// ============================================================================
// Platform-specific image operations (expect declarations)
// ============================================================================

/**
 * Resizes a platform image to the specified dimensions.
 *
 * @param image The source image
 * @param width Target width in pixels
 * @param height Target height in pixels
 * @param interpolation The interpolation method to use
 * @return A new resized image
 */
public expect fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage

/**
 * Crops a platform image to the specified region.
 *
 * @param image The source image
 * @param x Left edge of crop region
 * @param y Top edge of crop region
 * @param width Width of crop region
 * @param height Height of crop region
 * @return A new cropped image
 */
public expect fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage

/**
 * Rotates a platform image by the specified degrees.
 *
 * @param image The source image
 * @param degrees Rotation angle in degrees (positive = clockwise)
 * @param interpolation The interpolation method to use
 * @return A new rotated image
 */
public expect fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage

/**
 * Pads a platform image with the specified margins.
 *
 * @param image The source image
 * @param top Padding at top edge
 * @param bottom Padding at bottom edge
 * @param left Padding at left edge
 * @param right Padding at right edge
 * @param red Red component of padding color (0-255)
 * @param green Green component of padding color (0-255)
 * @param blue Blue component of padding color (0-255)
 * @return A new padded image
 */
public expect fun padPlatformImage(
    image: PlatformBitmapImage,
    top: Int,
    bottom: Int,
    left: Int,
    right: Int,
    red: Int,
    green: Int,
    blue: Int
): PlatformBitmapImage

// ============================================================================
// Image Transform Classes (common implementation using expect functions)
// ============================================================================

/**
 * Resizes an image to the specified dimensions.
 *
 * ## Usage
 * ```kotlin
 * val resize = ImageResize(224, 224, Interpolation.BILINEAR)
 * val resized = resize.apply(originalImage)
 * ```
 *
 * @param width Target width in pixels
 * @param height Target height in pixels
 * @param interpolation The interpolation method (default: BILINEAR)
 */
public class ImageResize(
    public val width: Int,
    public val height: Int,
    public val interpolation: Interpolation = Interpolation.BILINEAR
) : Transform<PlatformBitmapImage, PlatformBitmapImage> {

    init {
        require(width > 0) { "Width must be positive: $width" }
        require(height > 0) { "Height must be positive: $height" }
    }

    override fun apply(input: PlatformBitmapImage): PlatformBitmapImage {
        return resizePlatformImage(input, width, height, interpolation)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        // Assumes input shape is (H, W) or (H, W, C) or (C, H, W)
        return when (inputShape.rank) {
            2 -> Shape(height, width)
            3 -> {
                // Determine if CHW or HWC based on typical channel counts
                if (inputShape[0] <= 4) {
                    // Likely CHW format
                    Shape(inputShape[0], height, width)
                } else {
                    // Likely HWC format
                    Shape(height, width, inputShape[2])
                }
            }
            4 -> Shape(inputShape[0], inputShape[1], height, width) // NCHW
            else -> inputShape
        }
    }

    override fun toString(): String = "ImageResize(width=$width, height=$height, interpolation=$interpolation)"
}

/**
 * Crops an image by removing pixels from the edges.
 *
 * ## Usage
 * ```kotlin
 * val crop = ImageCrop(top = 10, bottom = 10, left = 10, right = 10)
 * val cropped = crop.apply(originalImage)
 * ```
 *
 * @param top Pixels to remove from top
 * @param bottom Pixels to remove from bottom
 * @param left Pixels to remove from left
 * @param right Pixels to remove from right
 */
public class ImageCrop(
    public val top: Int = 0,
    public val bottom: Int = 0,
    public val left: Int = 0,
    public val right: Int = 0
) : Transform<PlatformBitmapImage, PlatformBitmapImage> {

    init {
        require(top >= 0) { "Top must be non-negative: $top" }
        require(bottom >= 0) { "Bottom must be non-negative: $bottom" }
        require(left >= 0) { "Left must be non-negative: $left" }
        require(right >= 0) { "Right must be non-negative: $right" }
    }

    override fun apply(input: PlatformBitmapImage): PlatformBitmapImage {
        val (width, height) = platformImageSize(input)
        val newWidth = width - left - right
        val newHeight = height - top - bottom

        require(newWidth > 0) { "Crop would result in zero or negative width: $newWidth" }
        require(newHeight > 0) { "Crop would result in zero or negative height: $newHeight" }

        return cropPlatformImage(input, left, top, newWidth, newHeight)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        // This is a simplification - actual output depends on input image size
        return inputShape
    }

    override fun toString(): String = "ImageCrop(top=$top, bottom=$bottom, left=$left, right=$right)"
}

/**
 * Center crops an image to a square of the specified size.
 *
 * ## Usage
 * ```kotlin
 * val centerCrop = ImageCenterCrop(224)
 * val cropped = centerCrop.apply(originalImage)
 * ```
 *
 * @param size The size of the output square (width and height)
 */
public class ImageCenterCrop(
    public val size: Int
) : Transform<PlatformBitmapImage, PlatformBitmapImage> {

    init {
        require(size > 0) { "Size must be positive: $size" }
    }

    override fun apply(input: PlatformBitmapImage): PlatformBitmapImage {
        val (width, height) = platformImageSize(input)

        require(width >= size) { "Image width ($width) must be >= crop size ($size)" }
        require(height >= size) { "Image height ($height) must be >= crop size ($size)" }

        val left = (width - size) / 2
        val top = (height - size) / 2

        return cropPlatformImage(input, left, top, size, size)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        return when (inputShape.rank) {
            2 -> Shape(size, size)
            3 -> {
                if (inputShape[0] <= 4) {
                    Shape(inputShape[0], size, size) // CHW
                } else {
                    Shape(size, size, inputShape[2]) // HWC
                }
            }
            4 -> Shape(inputShape[0], inputShape[1], size, size) // NCHW
            else -> inputShape
        }
    }

    override fun toString(): String = "ImageCenterCrop(size=$size)"
}

/**
 * Rotates an image by the specified degrees.
 *
 * ## Usage
 * ```kotlin
 * val rotate = ImageRotate(90f)
 * val rotated = rotate.apply(originalImage)
 * ```
 *
 * @param degrees Rotation angle (positive = clockwise)
 * @param interpolation The interpolation method (default: BILINEAR)
 */
public class ImageRotate(
    public val degrees: Float,
    public val interpolation: Interpolation = Interpolation.BILINEAR
) : Transform<PlatformBitmapImage, PlatformBitmapImage> {

    override fun apply(input: PlatformBitmapImage): PlatformBitmapImage {
        return rotatePlatformImage(input, degrees, interpolation)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        // Rotation may change dimensions for non-90-degree angles
        // For simplicity, assume shape is preserved
        return inputShape
    }

    override fun toString(): String = "ImageRotate(degrees=$degrees, interpolation=$interpolation)"
}

/**
 * Pads an image by adding pixels to the edges.
 *
 * ## Usage
 * ```kotlin
 * val pad = ImagePad(top = 10, bottom = 10, left = 10, right = 10)
 * val padded = pad.apply(originalImage)
 * ```
 *
 * @param top Pixels to add at top
 * @param bottom Pixels to add at bottom
 * @param left Pixels to add at left
 * @param right Pixels to add at right
 * @param red Red component of padding color (0-255)
 * @param green Green component of padding color (0-255)
 * @param blue Blue component of padding color (0-255)
 */
public class ImagePad(
    public val top: Int = 0,
    public val bottom: Int = 0,
    public val left: Int = 0,
    public val right: Int = 0,
    public val red: Int = 0,
    public val green: Int = 0,
    public val blue: Int = 0
) : Transform<PlatformBitmapImage, PlatformBitmapImage> {

    init {
        require(top >= 0) { "Top must be non-negative: $top" }
        require(bottom >= 0) { "Bottom must be non-negative: $bottom" }
        require(left >= 0) { "Left must be non-negative: $left" }
        require(right >= 0) { "Right must be non-negative: $right" }
        require(red in 0..255) { "Red must be in 0..255: $red" }
        require(green in 0..255) { "Green must be in 0..255: $green" }
        require(blue in 0..255) { "Blue must be in 0..255: $blue" }
    }

    override fun apply(input: PlatformBitmapImage): PlatformBitmapImage {
        return padPlatformImage(input, top, bottom, left, right, red, green, blue)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        // This is a simplification - actual output depends on input image size
        return inputShape
    }

    override fun toString(): String = "ImagePad(top=$top, bottom=$bottom, left=$left, right=$right, color=($red,$green,$blue))"
}

/**
 * Converts a platform image to a tensor.
 *
 * Uses SKaiNET's `platformImageToArgb` function to convert the image to a
 * tensor with shape (1, 3, H, W) and float values in [0, 255] range.
 *
 * ## Usage
 * ```kotlin
 * val toTensor = ImageToTensor(executionContext)
 * val tensor = toTensor.apply(image)  // Shape: (1, 3, H, W)
 * ```
 *
 * @param ctx The execution context for tensor creation
 */
public class ImageToTensor(
    private val ctx: ExecutionContext
) : Transform<PlatformBitmapImage, Tensor<FP16, Float>> {

    override fun apply(input: PlatformBitmapImage): Tensor<FP16, Float> {
        return platformImageToArgb(input, ctx)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        // Output is always (1, 3, H, W) from platformImageToArgb
        return when (inputShape.rank) {
            2 -> Shape(1, 3, inputShape[0], inputShape[1]) // (H, W) -> (1, 3, H, W)
            3 -> {
                if (inputShape[0] <= 4) {
                    Shape(1, inputShape[0], inputShape[1], inputShape[2]) // CHW -> NCHW
                } else {
                    Shape(1, inputShape[2], inputShape[0], inputShape[1]) // HWC -> NCHW
                }
            }
            else -> inputShape
        }
    }

    override fun toString(): String = "ImageToTensor"
}
