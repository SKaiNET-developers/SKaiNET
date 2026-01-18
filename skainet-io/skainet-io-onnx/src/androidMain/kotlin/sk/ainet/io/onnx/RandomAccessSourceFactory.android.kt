package sk.ainet.io.onnx

import sk.ainet.io.RandomAccessSource

/**
 * Android implementation of [createOnnxRandomAccessSource].
 *
 * Returns null on Android as file access patterns differ.
 * Callers should fall back to standard ONNX loading.
 *
 * Future: Could implement using Android-specific file APIs.
 */
public actual fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? = null
