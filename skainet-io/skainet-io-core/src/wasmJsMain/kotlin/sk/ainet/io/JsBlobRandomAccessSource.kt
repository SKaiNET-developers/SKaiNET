@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class)

package sk.ainet.io

import kotlinx.coroutines.await
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.Int8Array
import org.khronos.webgl.get
import org.w3c.files.Blob
import kotlin.js.Promise

@JsFun("(msg) => console.log(msg)")
private external fun consoleLog(msg: String)

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
    private val blob: Blob,
    private val preloadedBuffer: ByteArray,
    private val blobSize: Long
) : RandomAccessSource, SuspendingRandomAccessSource {

    override val size: Long = blobSize

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
    /**
     * The suspending read (#1037): unlike [readAt], this is not limited to the preloaded window —
     * a range outside it is fetched from the blob instead of failing.
     */
    override suspend fun read(position: Long, length: Int): ByteArray = readAtAsync(position, length)

    /** Suspending read into [buffer]; see [readAt]. */
    override suspend fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        val bytes = readAtAsync(position, length)
        bytes.copyInto(buffer, offset)
        return bytes.size
    }

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

        // Read from blob using slice + arrayBuffer via JS interop
        val slice = sliceBlob(blob, pos, requestedEnd)
        val arrayBuffer = blobArrayBuffer(slice).await<ArrayBuffer>()
        return arrayBufferToByteArray(arrayBuffer)
    }

    public companion object {
        /**
         * Default preload size: 50MB should cover metadata for most models.
         */
        public const val DEFAULT_PRELOAD_SIZE: Int = 50 * 1024 * 1024

        /**
         * Open a Blob for streaming random access.
         *
         * @param blob A browser Blob or File object
         * @param preloadSize Number of bytes to preload for synchronous access
         */
        public suspend fun open(blob: Blob, preloadSize: Int = DEFAULT_PRELOAD_SIZE): JsBlobRandomAccessSource {
            val blobSize = getBlobSize(blob)
            consoleLog("[JsBlobRAS] blobSize=$blobSize, preloadSize=$preloadSize")
            val actualPreloadSize = minOf(preloadSize, blobSize)
            consoleLog("[JsBlobRAS] actualPreloadSize=$actualPreloadSize")
            val slice = sliceBlob(blob, 0, actualPreloadSize)
            val sliceSize = getBlobSize(slice)
            consoleLog("[JsBlobRAS] sliceSize=$sliceSize")
            val arrayBuffer = blobArrayBuffer(slice).await<ArrayBuffer>()
            val bufferByteLength = getArrayBufferByteLength(arrayBuffer)
            consoleLog("[JsBlobRAS] arrayBuffer.byteLength=$bufferByteLength")
            val bytes = arrayBufferToByteArray(arrayBuffer)
            consoleLog("[JsBlobRAS] bytes.size=${bytes.size}")
            // Log first 8 bytes as hex for ONNX magic check
            if (bytes.size >= 8) {
                val hex = bytes.take(8).joinToString(" ") { (it.toInt() and 0xFF).toString(16).padStart(2, '0') }
                consoleLog("[JsBlobRAS] first 8 bytes: $hex")
            }
            return JsBlobRandomAccessSource(blob, bytes, blobSize.toLong())
        }

        /**
         * Convert JavaScript ArrayBuffer to Kotlin ByteArray using Int8Array.
         */
        private fun arrayBufferToByteArray(buffer: ArrayBuffer): ByteArray {
            val int8Array = Int8Array(buffer)
            val length = int8Array.length
            consoleLog("[JsBlobRAS] converting $length bytes via Int8Array")
            val bytes = ByteArray(length) { i -> int8Array[i] }
            // Debug: log first 16 bytes
            if (length >= 16) {
                val first16 = (0 until 16).map { int8Array[it].toInt() and 0xFF }
                consoleLog("[JsBlobRAS] first 16 raw values: $first16")
            }
            return bytes
        }
    }
}

// Use external JS function declarations for proper WASM interop
@JsFun("(blob) => blob.size")
private external fun getBlobSize(blob: Blob): Int

@JsFun("(blob) => blob.arrayBuffer()")
private external fun blobArrayBuffer(blob: Blob): Promise<ArrayBuffer>

@JsFun("(buf) => buf.byteLength")
private external fun getArrayBufferByteLength(buf: ArrayBuffer): Int

@JsFun("(blob, start, end) => blob.slice(start, end)")
private external fun sliceBlob(blob: Blob, start: Int, end: Int): Blob
