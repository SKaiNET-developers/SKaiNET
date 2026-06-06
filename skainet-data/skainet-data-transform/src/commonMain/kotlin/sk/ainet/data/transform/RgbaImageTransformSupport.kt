package sk.ainet.data.transform

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sin

internal data class PackedRgbaImage(
    val width: Int,
    val height: Int,
    val rgba: ByteArray
) {
    init {
        require(width > 0) { "Width must be positive: $width" }
        require(height > 0) { "Height must be positive: $height" }
        require(rgba.size == width * height * 4) {
            "RGBA buffer must contain width * height * 4 bytes, got ${rgba.size} for ${width}x$height"
        }
    }
}

internal fun cropPackedRgbaImage(
    image: PackedRgbaImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PackedRgbaImage {
    val out = ByteArray(width * height * 4)
    for (row in 0 until height) {
        for (col in 0 until width) {
            copyPixel(
                image.rgba,
                rgbaIndex(image.width, x + col, y + row),
                out,
                rgbaIndex(width, col, row)
            )
        }
    }
    return PackedRgbaImage(width, height, out)
}

internal fun padPackedRgbaImage(
    image: PackedRgbaImage,
    top: Int,
    bottom: Int,
    left: Int,
    right: Int,
    red: Int,
    green: Int,
    blue: Int
): PackedRgbaImage {
    val width = image.width + left + right
    val height = image.height + top + bottom
    val out = ByteArray(width * height * 4)

    for (y in 0 until height) {
        for (x in 0 until width) {
            val dst = rgbaIndex(width, x, y)
            out[dst] = red.toByte()
            out[dst + 1] = green.toByte()
            out[dst + 2] = blue.toByte()
            out[dst + 3] = 0xFF.toByte()
        }
    }

    for (y in 0 until image.height) {
        for (x in 0 until image.width) {
            copyPixel(
                image.rgba,
                rgbaIndex(image.width, x, y),
                out,
                rgbaIndex(width, x + left, y + top)
            )
        }
    }

    return PackedRgbaImage(width, height, out)
}

internal fun resizePackedRgbaImage(
    image: PackedRgbaImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PackedRgbaImage {
    if (image.width == width && image.height == height) {
        return PackedRgbaImage(width, height, image.rgba.copyOf())
    }

    val out = ByteArray(width * height * 4)
    for (y in 0 until height) {
        val srcY = ((y + 0.5) * image.height / height) - 0.5
        for (x in 0 until width) {
            val srcX = ((x + 0.5) * image.width / width) - 0.5
            val dst = rgbaIndex(width, x, y)
            when (interpolation) {
                Interpolation.NEAREST -> sampleNearestClamped(image, srcX, srcY, out, dst)
                Interpolation.BILINEAR, Interpolation.BICUBIC -> sampleBilinearClamped(image, srcX, srcY, out, dst)
            }
        }
    }

    return PackedRgbaImage(width, height, out)
}

internal fun rotatePackedRgbaImage(
    image: PackedRgbaImage,
    degrees: Float,
    interpolation: Interpolation
): PackedRgbaImage {
    val radians = degrees * PI / 180.0
    val absSin = abs(sin(radians))
    val absCos = abs(cos(radians))
    val width = ceil(image.width * absCos + image.height * absSin).toInt().coerceAtLeast(1)
    val height = ceil(image.width * absSin + image.height * absCos).toInt().coerceAtLeast(1)
    val out = ByteArray(width * height * 4)

    val srcCx = (image.width - 1) / 2.0
    val srcCy = (image.height - 1) / 2.0
    val dstCx = (width - 1) / 2.0
    val dstCy = (height - 1) / 2.0
    val cosTheta = cos(radians)
    val sinTheta = sin(radians)

    for (y in 0 until height) {
        for (x in 0 until width) {
            val dx = x - dstCx
            val dy = y - dstCy
            val srcX = dx * cosTheta - dy * sinTheta + srcCx
            val srcY = dx * sinTheta + dy * cosTheta + srcCy
            val dst = rgbaIndex(width, x, y)
            when (interpolation) {
                Interpolation.NEAREST -> sampleNearestTransparent(image, srcX, srcY, out, dst)
                Interpolation.BILINEAR, Interpolation.BICUBIC -> sampleBilinearTransparent(image, srcX, srcY, out, dst)
            }
        }
    }

    return PackedRgbaImage(width, height, out)
}

private fun sampleNearestClamped(
    image: PackedRgbaImage,
    srcX: Double,
    srcY: Double,
    out: ByteArray,
    dstIndex: Int
) {
    val x = srcX.roundToInt().coerceIn(0, image.width - 1)
    val y = srcY.roundToInt().coerceIn(0, image.height - 1)
    copyPixel(image.rgba, rgbaIndex(image.width, x, y), out, dstIndex)
}

private fun sampleNearestTransparent(
    image: PackedRgbaImage,
    srcX: Double,
    srcY: Double,
    out: ByteArray,
    dstIndex: Int
) {
    val x = srcX.roundToInt()
    val y = srcY.roundToInt()
    if (x !in 0 until image.width || y !in 0 until image.height) {
        clearPixel(out, dstIndex)
        return
    }
    copyPixel(image.rgba, rgbaIndex(image.width, x, y), out, dstIndex)
}

private fun sampleBilinearClamped(
    image: PackedRgbaImage,
    srcX: Double,
    srcY: Double,
    out: ByteArray,
    dstIndex: Int
) {
    for (channel in 0 until 4) {
        out[dstIndex + channel] = bilinearChannel(image, srcX, srcY, channel, false).toByte()
    }
}

private fun sampleBilinearTransparent(
    image: PackedRgbaImage,
    srcX: Double,
    srcY: Double,
    out: ByteArray,
    dstIndex: Int
) {
    for (channel in 0 until 4) {
        out[dstIndex + channel] = bilinearChannel(image, srcX, srcY, channel, true).toByte()
    }
}

private fun bilinearChannel(
    image: PackedRgbaImage,
    srcX: Double,
    srcY: Double,
    channel: Int,
    transparentOutside: Boolean
): Int {
    val x0 = floor(srcX).toInt()
    val y0 = floor(srcY).toInt()
    val x1 = x0 + 1
    val y1 = y0 + 1
    val fx = srcX - x0
    val fy = srcY - y0

    val c00 = sampleChannel(image, x0, y0, channel, transparentOutside)
    val c10 = sampleChannel(image, x1, y0, channel, transparentOutside)
    val c01 = sampleChannel(image, x0, y1, channel, transparentOutside)
    val c11 = sampleChannel(image, x1, y1, channel, transparentOutside)

    val top = c00 * (1.0 - fx) + c10 * fx
    val bottom = c01 * (1.0 - fx) + c11 * fx
    return (top * (1.0 - fy) + bottom * fy).roundToInt().coerceIn(0, 255)
}

private fun sampleChannel(
    image: PackedRgbaImage,
    x: Int,
    y: Int,
    channel: Int,
    transparentOutside: Boolean
): Double {
    if (x !in 0 until image.width || y !in 0 until image.height) {
        return if (transparentOutside) 0.0 else {
            val clampedX = x.coerceIn(0, image.width - 1)
            val clampedY = y.coerceIn(0, image.height - 1)
            image.rgba[rgbaIndex(image.width, clampedX, clampedY) + channel].toUByte().toInt().toDouble()
        }
    }
    return image.rgba[rgbaIndex(image.width, x, y) + channel].toUByte().toInt().toDouble()
}

private fun copyPixel(src: ByteArray, srcIndex: Int, dst: ByteArray, dstIndex: Int) {
    dst[dstIndex] = src[srcIndex]
    dst[dstIndex + 1] = src[srcIndex + 1]
    dst[dstIndex + 2] = src[srcIndex + 2]
    dst[dstIndex + 3] = src[srcIndex + 3]
}

private fun clearPixel(dst: ByteArray, dstIndex: Int) {
    dst[dstIndex] = 0
    dst[dstIndex + 1] = 0
    dst[dstIndex + 2] = 0
    dst[dstIndex + 3] = 0
}

private fun rgbaIndex(width: Int, x: Int, y: Int): Int = ((y * width) + x) * 4
