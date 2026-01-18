package sk.ainet.io.onnx

import sk.ainet.io.RandomAccessSource

/**
 * Native implementation of [createOnnxRandomAccessSource].
 *
 * Returns null as native random file access is not yet implemented.
 * Callers should fall back to legacy OnnxLoader which loads the full file.
 *
 * Future: Could implement using POSIX pread() for efficient random access.
 */
public actual fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? = null
