package sk.ainet.io.image

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

public actual class PlatformBitmapImage public constructor(
    public val width: Int,
    public val height: Int,
    rgba: ByteArray
) {
    public val rgba: ByteArray = rgba.copyOf()

    init {
        require(this.rgba.size == width * height * 4) {
            "RGBA buffer must contain width * height * 4 bytes, got ${this.rgba.size} for ${width}x$height"
        }
    }
}

public actual fun platformImageToArgb(image: PlatformBitmapImage, ctx: ExecutionContext): Tensor<FP16, Float> {
    return packedRgbaToTensor(PackedRgbaImage(image.width, image.height, image.rgba), ctx)
}

public actual fun argbToPlatformImage(image: Tensor<FP16, Float>, ctx: ExecutionContext): PlatformBitmapImage {
    val packed = tensorToPackedRgba(image)
    return PlatformBitmapImage(packed.width, packed.height, packed.rgba)
}

public actual fun platformImageToRgbByteArray(image: PlatformBitmapImage): ByteArray {
    return rgbByteArrayFromPackedRgba(PackedRgbaImage(image.width, image.height, image.rgba))
}

public actual fun platformImageSize(image: PlatformBitmapImage): Pair<Int, Int> {
    return image.width to image.height
}
