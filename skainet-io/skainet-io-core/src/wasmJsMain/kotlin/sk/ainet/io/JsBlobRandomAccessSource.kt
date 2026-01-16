package sk.ainet.io

import kotlinx.coroutines.await
import kotlin.js.Promise
import kotlin.js.JsAny

/**
 * WASM JS implementation of [RandomAccessSource] using Blob API.
 *
 * Uses Blob.slice() for efficient random access to file data without loading
 * the entire file into memory.
 *
 * **Important**: Browser Blob operations are inherently asynchronous.
 * This implementation pre-loads metadata section during [open] for synchronous
 * access via [readAt], and provides [readAtAsync] for accessing data beyond
 * the preloaded buffer.
 */
public class JsBlobRandomAccessSource private constructor(
    private val blob: JsAny, // Browser Blob object
    private val preloadedBuffer: ByteArray
) : RandomAccessSource {

    override val size: Long = getBlobSize(blob).toLong()

    /**
     * The size of the preloaded buffer in bytes.
     * Reads within this range are synchronous.
     */
    public val preloadedSize: Int = preloadedBuffer.size

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }

        if (length == 0) return ByteArray(0)

        val pos = position.toInt()
        val requestedEnd = pos + length

        // If within preloaded buffer, return from cache
        if (requestedEnd <= preloadedBuffer.size) {
            return preloadedBuffer.copyOfRange(pos, requestedEnd)
        }

        // Beyond preloaded data - throw helpful error
        throw UnsupportedOperationException(
            "Synchronous read beyond preloaded buffer (${preloadedBuffer.size} bytes). " +
            "Position: $position, Length: $length. " +
            "Use readAtAsync() for tensor data loading, or increase preloadSize when opening."
        )
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }

        if (length == 0) return 0

        val data = readAt(position, length)
        data.copyInto(buffer, offset, 0, data.size)
        return data.size
    }

    override fun close() {
        // Blob doesn't need explicit closing
    }

    /**
     * Asynchronously read bytes from any position.
     *
     * Use this for loading tensor data that may be beyond the preloaded buffer.
     */
    public suspend fun readAtAsync(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }

        if (length == 0) return ByteArray(0)

        val pos = position.toInt()
        val requestedEnd = pos + length

        // If within preloaded buffer, return from cache (fast path)
        if (requestedEnd <= preloadedBuffer.size) {
            return preloadedBuffer.copyOfRange(pos, requestedEnd)
        }

        // Read from blob using slice + arrayBuffer
        val slice = blobSlice(blob, pos, requestedEnd)
        val arrayBuffer = blobToArrayBuffer(slice).await<JsAny>()
        return jsArrayBufferToByteArray(arrayBuffer)
    }

    public companion object {
        /**
         * Default preload size: 50MB should cover metadata for most models.
         */
        public const val DEFAULT_PRELOAD_SIZE: Int = 50 * 1024 * 1024

        /**
         * Open a Blob for streaming random access.
         *
         * @param blob A browser Blob or File object (passed as JsAny)
         * @param preloadSize Number of bytes to preload for synchronous access
         */
        public suspend fun open(blob: JsAny, preloadSize: Int = DEFAULT_PRELOAD_SIZE): JsBlobRandomAccessSource {
            val blobSize = getBlobSize(blob)
            val actualPreloadSize = minOf(preloadSize, blobSize)
            val slice = blobSlice(blob, 0, actualPreloadSize)
            val arrayBuffer = blobToArrayBuffer(slice).await<JsAny>()
            val bytes = jsArrayBufferToByteArray(arrayBuffer)
            return JsBlobRandomAccessSource(blob, bytes)
        }
    }
}

// External JS helper functions using @JsFun for WASM
@JsFun("(blob) => blob.size")
private external fun getBlobSize(blob: JsAny): Int

@JsFun("(blob, start, end) => blob.slice(start, end)")
private external fun blobSlice(blob: JsAny, start: Int, end: Int): JsAny

@JsFun("(blob) => blob.arrayBuffer()")
private external fun blobToArrayBuffer(blob: JsAny): Promise<JsAny>

@JsFun("""(ab) => {
    const view = new Uint8Array(ab);
    const len = view.length;
    const arr = new Int8Array(len);
    for (let i = 0; i < len; i++) {
        arr[i] = view[i] > 127 ? view[i] - 256 : view[i];
    }
    return arr;
}""")
private external fun jsArrayBufferToInt8Array(ab: JsAny): JsAny

@JsFun("(arr) => arr.length")
private external fun jsInt8ArrayLength(arr: JsAny): Int

@JsFun("(arr, i) => arr[i]")
private external fun jsInt8ArrayGet(arr: JsAny, i: Int): Byte

private fun jsArrayBufferToByteArray(ab: JsAny): ByteArray {
    val int8Array = jsArrayBufferToInt8Array(ab)
    val len = jsInt8ArrayLength(int8Array)
    return ByteArray(len) { jsInt8ArrayGet(int8Array, it) }
}
