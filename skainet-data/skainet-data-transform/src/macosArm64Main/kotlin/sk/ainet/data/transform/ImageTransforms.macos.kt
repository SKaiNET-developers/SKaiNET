/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

@file:OptIn(ExperimentalForeignApi::class)

package sk.ainet.data.transform

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.CValue
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import platform.AppKit.NSImage
import platform.CoreFoundation.CFDataCreate
import platform.CoreFoundation.kCFAllocatorDefault
import platform.CoreGraphics.CGBitmapContextCreate
import platform.CoreGraphics.CGColorRenderingIntent
import platform.CoreGraphics.CGColorSpaceCreateDeviceRGB
import platform.CoreGraphics.CGContextDrawImage
import platform.CoreGraphics.CGDataProviderCreateWithCFData
import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.CGImageCreate
import platform.CoreGraphics.CGImageGetHeight
import platform.CoreGraphics.CGImageGetWidth
import platform.CoreGraphics.CGRectMake
import platform.CoreGraphics.CGSize
import platform.CoreGraphics.CGSizeMake
import sk.ainet.io.image.PlatformBitmapImage
import sk.ainet.io.image.platformImageSize

public actual fun resizePlatformImage(
    image: PlatformBitmapImage,
    width: Int,
    height: Int,
    interpolation: Interpolation
): PlatformBitmapImage {
    val raster = macosImageToPackedRgbaImage(image)
    return packedRgbaImageToMacosImage(resizePackedRgbaImage(raster, width, height, interpolation))
}

public actual fun cropPlatformImage(
    image: PlatformBitmapImage,
    x: Int,
    y: Int,
    width: Int,
    height: Int
): PlatformBitmapImage {
    val raster = macosImageToPackedRgbaImage(image)
    return packedRgbaImageToMacosImage(cropPackedRgbaImage(raster, x, y, width, height))
}

public actual fun rotatePlatformImage(
    image: PlatformBitmapImage,
    degrees: Float,
    interpolation: Interpolation
): PlatformBitmapImage {
    val raster = macosImageToPackedRgbaImage(image)
    return packedRgbaImageToMacosImage(rotatePackedRgbaImage(raster, degrees, interpolation))
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
    val raster = macosImageToPackedRgbaImage(image)
    return packedRgbaImageToMacosImage(
        padPackedRgbaImage(raster, top, bottom, left, right, red, green, blue)
    )
}

private fun macosImageToPackedRgbaImage(image: NSImage): PackedRgbaImage {
    val (width, height) = platformImageSize(image)
    return PackedRgbaImage(width, height, drawImageIntoRgbaBuffer(image, width, height))
}

private fun packedRgbaImageToMacosImage(image: PackedRgbaImage): NSImage {
    val cgImage = createCgImageFromRgba(image.rgba, image.width, image.height)
    val size: CValue<CGSize> = CGSizeMake(image.width.toDouble(), image.height.toDouble())
    return NSImage(cGImage = cgImage, size = size)
}

private fun drawImageIntoRgbaBuffer(image: NSImage, width: Int, height: Int): ByteArray = memScoped {
    val cg = image.CGImageForProposedRect(null, null, null)
        ?: error("NSImage has no CGImage representation")
    val colorSpace = CGColorSpaceCreateDeviceRGB()
        ?: error("Failed to create RGB color space")
    val bytesPerRow = width * 4
    val bitmapInfo: UInt = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value
    val buffer = ByteArray(width * height * 4)
    buffer.usePinned { pinned ->
        val ctx = CGBitmapContextCreate(
            pinned.addressOf(0),
            width.convert(),
            height.convert(),
            8.convert(),
            bytesPerRow.convert(),
            colorSpace,
            bitmapInfo
        ) ?: error("Failed to create bitmap context")
        val rect = CGRectMake(0.0, 0.0, width.toDouble(), height.toDouble())
        CGContextDrawImage(ctx, rect, cg)
    }
    buffer
}

private fun createCgImageFromRgba(bytes: ByteArray, width: Int, height: Int) = memScoped {
    val colorSpace = CGColorSpaceCreateDeviceRGB()
        ?: error("Failed to create RGB color space")
    val bytesPerRow = width * 4
    val provider = bytes.usePinned { pinned ->
        val cfData = CFDataCreate(
            kCFAllocatorDefault,
            pinned.addressOf(0).reinterpret(),
            bytes.size.convert()
        ) ?: error("Failed to create CFData")
        CGDataProviderCreateWithCFData(cfData)
            ?: error("Failed to create CGDataProvider")
    }

    val bitmapInfo: UInt = CGImageAlphaInfo.kCGImageAlphaPremultipliedLast.value
    CGImageCreate(
        width.convert(),
        height.convert(),
        8.convert(),
        32.convert(),
        bytesPerRow.convert(),
        colorSpace,
        bitmapInfo,
        provider,
        null,
        true,
        CGColorRenderingIntent.kCGRenderingIntentDefault
    ) ?: error("Failed to create CGImage")
}
