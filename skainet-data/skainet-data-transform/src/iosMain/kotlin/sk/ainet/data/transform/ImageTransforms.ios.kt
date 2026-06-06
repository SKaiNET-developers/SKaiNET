/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.io.image.PlatformBitmapImage

private fun PlatformBitmapImage.toPackedRgbaImage(): PackedRgbaImage =
    PackedRgbaImage(width, height, rgba)

private fun PackedRgbaImage.toPlatformBitmapImage(): PlatformBitmapImage =
    PlatformBitmapImage(width, height, rgba)

public actual fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage {
    return resizePackedRgbaImage(image.toPackedRgbaImage(), width, height, interpolation).toPlatformBitmapImage()
}

public actual fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage {
    return cropPackedRgbaImage(image.toPackedRgbaImage(), x, y, width, height).toPlatformBitmapImage()
}

public actual fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage {
    return rotatePackedRgbaImage(image.toPackedRgbaImage(), degrees, interpolation).toPlatformBitmapImage()
}

public actual fun padPlatformImage(
    image: PlatformBitmapImage,
    top: Int,
    bottom: Int,
    left: Int,
    right: Int,
    red: Int,
    green: Int,
    blue: Int
): PlatformBitmapImage {
    return padPackedRgbaImage(
        image.toPackedRgbaImage(),
        top,
        bottom,
        left,
        right,
        red,
        green,
        blue
    ).toPlatformBitmapImage()
}
