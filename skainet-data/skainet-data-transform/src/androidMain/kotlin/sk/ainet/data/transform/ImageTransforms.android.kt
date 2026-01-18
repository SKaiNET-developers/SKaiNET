/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import sk.ainet.io.image.PlatformBitmapImage

/**
 * Android implementation of image resize using Bitmap APIs.
 */
public actual fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage {
    val filter = interpolation != Interpolation.NEAREST
    return Bitmap.createScaledBitmap(image, width, height, filter)
}

/**
 * Android implementation of image crop using Bitmap APIs.
 */
public actual fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage {
    return Bitmap.createBitmap(image, x, y, width, height)
}

/**
 * Android implementation of image rotation using Matrix.
 */
public actual fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage {
    val matrix = Matrix()
    matrix.postRotate(degrees)

    val filter = interpolation != Interpolation.NEAREST
    return Bitmap.createBitmap(image, 0, 0, image.width, image.height, matrix, filter)
}

/**
 * Android implementation of image padding using Canvas.
 */
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
    val newWidth = image.width + left + right
    val newHeight = image.height + top + bottom

    val output = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)

    // Fill with padding color
    val color = android.graphics.Color.rgb(red, green, blue)
    canvas.drawColor(color)

    // Draw original image at offset position
    canvas.drawBitmap(image, left.toFloat(), top.toFloat(), null)

    return output
}
