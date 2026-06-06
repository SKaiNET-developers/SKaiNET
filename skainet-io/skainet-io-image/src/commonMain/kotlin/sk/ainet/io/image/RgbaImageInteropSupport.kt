package sk.ainet.io.image

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP16

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

internal fun packedRgbaToTensor(image: PackedRgbaImage, ctx: ExecutionContext): Tensor<FP16, Float> =
    data<FP16, Float>(ctx) {
        val rgbChw = FloatArray(image.width * image.height * 3)
        var p = 0
        for (y in 0 until image.height) {
            for (x in 0 until image.width) {
                val hw = y * image.width + x
                rgbChw[hw] = image.rgba[p].toUByte().toInt().toFloat()
                rgbChw[image.height * image.width + hw] = image.rgba[p + 1].toUByte().toInt().toFloat()
                rgbChw[2 * image.height * image.width + hw] = image.rgba[p + 2].toUByte().toInt().toFloat()
                p += 4
            }
        }

        tensor<FP16, Float> {
            shape(1, 3, image.height, image.width) {
                fromArray(rgbChw)
            }
        }
    }

internal fun tensorToPackedRgba(image: Tensor<FP16, Float>): PackedRgbaImage {
    val shape = image.data.shape
    val channels = shape[1]
    val height = shape[2]
    val width = shape[3]
    val rgba = ByteArray(width * height * 4)

    var p = 0
    for (y in 0 until height) {
        for (x in 0 until width) {
            when (channels) {
                1 -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    rgba[p++] = v
                    rgba[p++] = v
                    rgba[p++] = v
                    rgba[p++] = 0xFF.toByte()
                }

                3 -> {
                    rgba[p++] = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    rgba[p++] = image.data[0, 1, y, x].toInt().coerceIn(0, 255).toByte()
                    rgba[p++] = image.data[0, 2, y, x].toInt().coerceIn(0, 255).toByte()
                    rgba[p++] = 0xFF.toByte()
                }

                else -> {
                    val v = image.data[0, 0, y, x].toInt().coerceIn(0, 255).toByte()
                    rgba[p++] = v
                    rgba[p++] = v
                    rgba[p++] = v
                    rgba[p++] = 0xFF.toByte()
                }
            }
        }
    }

    return PackedRgbaImage(width, height, rgba)
}

internal fun rgbByteArrayFromPackedRgba(image: PackedRgbaImage): ByteArray {
    val rgb = ByteArray(image.width * image.height * 3)
    var src = 0
    var dst = 0
    while (src < image.rgba.size) {
        rgb[dst++] = image.rgba[src]
        rgb[dst++] = image.rgba[src + 1]
        rgb[dst++] = image.rgba[src + 2]
        src += 4
    }
    return rgb
}
