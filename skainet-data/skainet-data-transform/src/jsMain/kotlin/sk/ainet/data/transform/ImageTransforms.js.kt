/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.io.image.PlatformBitmapImage

public actual fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage {
    throw NotImplementedError("Image transforms not yet implemented for JavaScript")
}

public actual fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage {
    throw NotImplementedError("Image transforms not yet implemented for JavaScript")
}

public actual fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage {
    throw NotImplementedError("Image transforms not yet implemented for JavaScript")
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
    throw NotImplementedError("Image transforms not yet implemented for JavaScript")
}
