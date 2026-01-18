package sk.ainet.app.notebook.tools

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import java.awt.image.BufferedImage

public enum class Layout {
    NCHW,
    NHWC,
    CHW,
    HWC,
    HW
}

public fun <T: DType, V> Tensor<T, V>.toImage(memoryLayout: Layout = Layout.HWC): BufferedImage {
    val dims = this.shape.dimensions

    // Determine width, height, channels, and an indexer based on layout
    var height: Int
    var width: Int
    var channels: Int
    val indexer: (Int, Int, Int) -> Any? = when (memoryLayout) {
        Layout.HWC -> {
            require(dims.size == 3) { "HWC expects rank 3 tensor (H,W,C), got rank ${dims.size} with shape ${this.shape}" }
            height = dims[0]
            width = dims[1]
            channels = dims[2]
            { y: Int, x: Int, ch: Int -> this.data[y, x, ch] }
        }
        Layout.HW -> {
            require(dims.size == 2) { "HW expects rank 2 tensor (H,W), got rank ${dims.size} with shape ${this.shape}" }
            height = dims[0]
            width = dims[1]
            channels = 1
            { y: Int, x: Int, _: Int -> this.data[y, x] }
        }
        Layout.CHW -> {
            require(dims.size == 3) { "CHW expects rank 3 tensor (C,H,W), got rank ${dims.size} with shape ${this.shape}" }
            channels = dims[0]
            height = dims[1]
            width = dims[2]
            { y: Int, x: Int, ch: Int -> this.data[ch, y, x] }
        }
        Layout.NHWC -> {
            require(dims.size == 4) { "NHWC expects rank 4 tensor (N,H,W,C), got rank ${dims.size} with shape ${this.shape}" }
            height = dims[1]
            width = dims[2]
            channels = dims[3]
            { y: Int, x: Int, ch: Int -> this.data[0, y, x, ch] }
        }
        Layout.NCHW -> {
            require(dims.size == 4) { "NCHW expects rank 4 tensor (N,C,H,W), got rank ${dims.size} with shape ${this.shape}" }
            channels = dims[1]
            height = dims[2]
            width = dims[3]
            { y: Int, x: Int, ch: Int -> this.data[0, ch, y, x] }
        }
    }

    require(channels in setOf(1, 3, 4)) { "Only 1 (grayscale), 3 (RGB), or 4 (RGBA) channels supported, got $channels" }

    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)

    // Heuristic: detect if values appear normalized [0,1] by sampling
    var sampleMin = Float.POSITIVE_INFINITY
    var sampleMax = Float.NEGATIVE_INFINITY
    val sampleY = 0
    val sampleX = 0
    val sampleCount = minOf(channels, 4)
    for (ch in 0 until sampleCount) {
        val v = valueToFloat(indexer(sampleY, sampleX, ch))
        if (v < sampleMin) sampleMin = v
        if (v > sampleMax) sampleMax = v
    }
    val multiply255 = sampleMax <= 1.0f && sampleMin >= 0.0f

    fun toByte(v: Float): Int {
        val scaled = if (multiply255) v * 255.0f else v
        val clamped = scaled.coerceIn(0.0f, 255.0f)
        return clamped.toInt()
    }

    for (y in 0 until height) {
        for (x in 0 until width) {
            val r: Int
            val g: Int
            val b: Int
            val a: Int
            if (channels == 1) {
                val v = toByte(valueToFloat(indexer(y, x, 0)))
                r = v; g = v; b = v; a = 255
            } else if (channels >= 3) {
                r = toByte(valueToFloat(indexer(y, x, 0)))
                g = toByte(valueToFloat(indexer(y, x, 1)))
                b = toByte(valueToFloat(indexer(y, x, 2)))
                a = if (channels >= 4) toByte(valueToFloat(indexer(y, x, 3))) else 255
            } else {
                // Fallback, should not happen due to require
                r = 0; g = 0; b = 0; a = 255
            }
            val argb = (a shl 24) or (r shl 16) or (g shl 8) or b
            image.setRGB(x, y, argb)
        }
    }

    return image
}

private fun valueToFloat(v: Any?): Float = when (v) {
    is Float -> v
    is Double -> v.toFloat()
    is Int -> v.toFloat()
    is Long -> v.toFloat()
    is Short -> v.toFloat()
    is Byte -> v.toFloat()
    is UByte -> v.toFloat()
    is UInt -> v.toFloat()
    is UShort -> v.toFloat()
    is ULong -> v.toFloat()
    is Boolean -> if (v) 1.0f else 0.0f
    else -> error("Unsupported tensor value type for image conversion: ${v?.let { it::class.simpleName } ?: "null"}")
}