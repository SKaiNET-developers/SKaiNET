/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.io.image.PlatformBitmapImage
import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage

/**
 * JVM implementation of image resize using Java2D.
 */
public actual fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage {
    val output = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val g2d: Graphics2D = output.createGraphics()

    // Set interpolation hint based on mode
    val interpolationHint = when (interpolation) {
        Interpolation.NEAREST -> RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
        Interpolation.BILINEAR -> RenderingHints.VALUE_INTERPOLATION_BILINEAR
        Interpolation.BICUBIC -> RenderingHints.VALUE_INTERPOLATION_BICUBIC
    }

    g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolationHint)
    g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
    g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    g2d.drawImage(image, 0, 0, width, height, null)
    g2d.dispose()

    return output
}

/**
 * JVM implementation of image crop using Java2D.
 */
public actual fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage {
    return image.getSubimage(x, y, width, height).let { subimage ->
        // getSubimage returns a view, so we need to copy to a new image
        val output = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        val g2d = output.createGraphics()
        g2d.drawImage(subimage, 0, 0, null)
        g2d.dispose()
        output
    }
}

/**
 * JVM implementation of image rotation using Java2D.
 */
public actual fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage {
    val radians = Math.toRadians(degrees.toDouble())
    val sin = kotlin.math.abs(kotlin.math.sin(radians))
    val cos = kotlin.math.abs(kotlin.math.cos(radians))

    val originalWidth = image.width
    val originalHeight = image.height

    // Calculate new dimensions to fit rotated image
    val newWidth = (originalWidth * cos + originalHeight * sin).toInt()
    val newHeight = (originalWidth * sin + originalHeight * cos).toInt()

    val output = BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB)
    val g2d: Graphics2D = output.createGraphics()

    // Set interpolation hint
    val interpolationHint = when (interpolation) {
        Interpolation.NEAREST -> RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
        Interpolation.BILINEAR -> RenderingHints.VALUE_INTERPOLATION_BILINEAR
        Interpolation.BICUBIC -> RenderingHints.VALUE_INTERPOLATION_BICUBIC
    }

    g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolationHint)
    g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
    g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

    // Create rotation transform centered on the new image
    val transform = AffineTransform()
    transform.translate(newWidth / 2.0, newHeight / 2.0)
    transform.rotate(radians)
    transform.translate(-originalWidth / 2.0, -originalHeight / 2.0)

    g2d.transform = transform
    g2d.drawImage(image, 0, 0, null)
    g2d.dispose()

    return output
}

/**
 * JVM implementation of image padding using Java2D.
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
    val originalWidth = image.width
    val originalHeight = image.height
    val newWidth = originalWidth + left + right
    val newHeight = originalHeight + top + bottom

    val output = BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB)
    val g2d: Graphics2D = output.createGraphics()

    // Fill with padding color
    g2d.color = Color(red, green, blue)
    g2d.fillRect(0, 0, newWidth, newHeight)

    // Draw original image at offset position
    g2d.drawImage(image, left, top, null)
    g2d.dispose()

    return output
}
