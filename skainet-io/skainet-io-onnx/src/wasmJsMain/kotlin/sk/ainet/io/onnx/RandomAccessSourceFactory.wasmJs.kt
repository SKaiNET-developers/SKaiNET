package sk.ainet.io.onnx

import org.w3c.files.Blob
import sk.ainet.io.JsBlobRandomAccessSource
import sk.ainet.io.RandomAccessSource

/**
 * WASM JS implementation of [createOnnxRandomAccessSource].
 *
 * Returns null for path-based access since file paths don't work in browsers.
 * Use [createOnnxRandomAccessSourceFromBlob] for browser file input.
 */
public actual fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? = null

/**
 * Create a RandomAccessSource from a browser Blob or File.
 *
 * This is the browser-specific way to create streaming ONNX readers.
 * Use with file input elements or File System Access API.
 *
 * @param blob The Blob or File to read from
 * @param preloadSize How much to preload for sync metadata access (default 50MB)
 * @return JsBlobRandomAccessSource for streaming access
 */
public suspend fun createOnnxRandomAccessSourceFromBlob(
    blob: Blob,
    preloadSize: Int = JsBlobRandomAccessSource.DEFAULT_PRELOAD_SIZE
): JsBlobRandomAccessSource {
    return JsBlobRandomAccessSource.open(blob, preloadSize)
}
