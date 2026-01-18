/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.context.ExecutionContext
import sk.ainet.io.image.PlatformBitmapImage
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

/**
 * DSL extensions for building image transformation pipelines.
 *
 * These extension functions provide a fluent API for composing image transforms:
 *
 * ```kotlin
 * val preprocessing = pipeline<PlatformBitmapImage>()
 *     .resize(224, 224)
 *     .centerCrop(224)
 *     .toTensor(ctx)
 *     .rescale(ctx, 255f)
 *     .normalize(ctx, ImageNet.mean, ImageNet.std)
 * ```
 */

/**
 * Chains an [ImageResize] transform.
 *
 * @param width Target width
 * @param height Target height
 * @param interpolation Interpolation method (default: BILINEAR)
 */
public fun <I> Transform<I, PlatformBitmapImage>.resize(
    width: Int,
    height: Int,
    interpolation: Interpolation = Interpolation.BILINEAR
): Transform<I, PlatformBitmapImage> = this then ImageResize(width, height, interpolation)

/**
 * Chains an [ImageCrop] transform that removes pixels from edges.
 *
 * @param top Pixels to remove from top
 * @param bottom Pixels to remove from bottom
 * @param left Pixels to remove from left
 * @param right Pixels to remove from right
 */
public fun <I> Transform<I, PlatformBitmapImage>.crop(
    top: Int = 0,
    bottom: Int = 0,
    left: Int = 0,
    right: Int = 0
): Transform<I, PlatformBitmapImage> = this then ImageCrop(top, bottom, left, right)

/**
 * Chains an [ImageCenterCrop] transform that extracts a centered square.
 *
 * @param size The size of the output square
 */
public fun <I> Transform<I, PlatformBitmapImage>.centerCrop(
    size: Int
): Transform<I, PlatformBitmapImage> = this then ImageCenterCrop(size)

/**
 * Chains an [ImageRotate] transform.
 *
 * @param degrees Rotation angle (positive = clockwise)
 * @param interpolation Interpolation method (default: BILINEAR)
 */
public fun <I> Transform<I, PlatformBitmapImage>.rotate(
    degrees: Float,
    interpolation: Interpolation = Interpolation.BILINEAR
): Transform<I, PlatformBitmapImage> = this then ImageRotate(degrees, interpolation)

/**
 * Chains an [ImagePad] transform that adds pixels to edges.
 *
 * @param top Pixels to add at top
 * @param bottom Pixels to add at bottom
 * @param left Pixels to add at left
 * @param right Pixels to add at right
 * @param red Red component of padding color (0-255)
 * @param green Green component of padding color (0-255)
 * @param blue Blue component of padding color (0-255)
 */
public fun <I> Transform<I, PlatformBitmapImage>.pad(
    top: Int = 0,
    bottom: Int = 0,
    left: Int = 0,
    right: Int = 0,
    red: Int = 0,
    green: Int = 0,
    blue: Int = 0
): Transform<I, PlatformBitmapImage> = this then ImagePad(top, bottom, left, right, red, green, blue)

/**
 * Chains an [ImageToTensor] transform that converts the image to a tensor.
 *
 * The output tensor has shape (1, 3, H, W) with float values in [0, 255].
 *
 * @param ctx The execution context for tensor creation
 */
public fun <I> Transform<I, PlatformBitmapImage>.toTensor(
    ctx: ExecutionContext
): Transform<I, Tensor<FP16, Float>> = this then ImageToTensor(ctx)

// ============================================================================
// Convenience functions for common preprocessing patterns
// ============================================================================

/**
 * Creates a standard ImageNet preprocessing pipeline.
 *
 * This pipeline:
 * 1. Resizes to 256x256
 * 2. Center crops to 224x224
 * 3. Converts to tensor
 * 4. Rescales to [0, 1]
 * 5. Normalizes with ImageNet statistics
 *
 * @param ctx The execution context
 * @return A transform that preprocesses images for ImageNet-trained models
 */
public fun imageNetPreprocessing(ctx: ExecutionContext): Transform<PlatformBitmapImage, Tensor<FP16, Float>> {
    return pipeline<PlatformBitmapImage>()
        .resize(256, 256)
        .centerCrop(224)
        .toTensor(ctx)
        .rescale(ctx, 255f)
        .normalize(ctx, ImageNet.mean, ImageNet.std)
}

/**
 * Creates a standard MNIST preprocessing pipeline.
 *
 * This pipeline:
 * 1. Resizes to 28x28
 * 2. Converts to tensor
 * 3. Rescales to [0, 1]
 * 4. Normalizes with MNIST statistics
 *
 * @param ctx The execution context
 * @return A transform that preprocesses images for MNIST models
 */
public fun mnistPreprocessing(ctx: ExecutionContext): Transform<PlatformBitmapImage, Tensor<FP16, Float>> {
    return pipeline<PlatformBitmapImage>()
        .resize(28, 28)
        .toTensor(ctx)
        .rescale(ctx, 255f)
        .normalize(ctx, MNISTNorm.mean, MNISTNorm.std)
}

/**
 * Creates a standard CIFAR-10 preprocessing pipeline.
 *
 * This pipeline:
 * 1. Resizes to 32x32
 * 2. Converts to tensor
 * 3. Rescales to [0, 1]
 * 4. Normalizes with CIFAR-10 statistics
 *
 * @param ctx The execution context
 * @return A transform that preprocesses images for CIFAR-10 models
 */
public fun cifar10Preprocessing(ctx: ExecutionContext): Transform<PlatformBitmapImage, Tensor<FP16, Float>> {
    return pipeline<PlatformBitmapImage>()
        .resize(32, 32)
        .toTensor(ctx)
        .rescale(ctx, 255f)
        .normalize(ctx, CIFAR10Norm.mean, CIFAR10Norm.std)
}
