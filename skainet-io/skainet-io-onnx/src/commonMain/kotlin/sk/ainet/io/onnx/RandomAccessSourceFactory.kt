package sk.ainet.io.onnx

import sk.ainet.io.RandomAccessSource

/**
 * Platform-specific factory for creating [RandomAccessSource] instances for ONNX files.
 *
 * Returns null on platforms that don't support random file access,
 * allowing callers to fall back to legacy sequential loading.
 *
 * Supported platforms:
 * - JVM: Uses FileChannel for efficient random access
 * - JS/Native: Returns null (use legacy OnnxLoader instead)
 *
 * @param filePath Path to the file
 * @return A RandomAccessSource, or null if not supported on this platform
 */
public expect fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource?
