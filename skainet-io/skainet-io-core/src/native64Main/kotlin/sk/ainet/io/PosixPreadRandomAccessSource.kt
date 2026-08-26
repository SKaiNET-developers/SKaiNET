package sk.ainet.io

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import platform.posix.O_RDONLY
import platform.posix.errno
import platform.posix.fstat
import platform.posix.pread
import platform.posix.stat
import platform.posix.strerror

/**
 * Native [RandomAccessSource] backed by POSIX `pread(2)`.
 *
 * `pread` is positional and atomic — it does not advance any shared seek
 * pointer — so concurrent reads from different positions are safe without
 * locking. [close] is single-shot.
 *
 * Used on macOS, iOS, and Linux native targets (which all share the
 * `nativeMain` source set in this module). Android uses a separate JNI
 * actual; JS / Wasm don't have a viable `pread` equivalent and continue
 * to fall back to the legacy GGUF reader.
 */
@OptIn(ExperimentalForeignApi::class)
public class PosixPreadRandomAccessSource private constructor(
    private val fd: Int,
    override val size: Long,
    /** The file these bytes come from — what `WeightResidency.MAPPED` would map (#1037, #1159). */
    override val filePath: String? = null,
) : RandomAccessSource {

    private var closed = false

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }
        if (length == 0) return ByteArray(0)

        val buffer = ByteArray(length)
        val bytesRead = readAt(position, buffer, 0, length)
        return if (bytesRead < length) buffer.copyOf(bytesRead) else buffer
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }
        check(!closed) { "Source is closed" }
        if (length == 0) return 0

        return buffer.usePinned { pinned ->
            var totalRead = 0
            while (totalRead < length) {
                val n = pread(
                    fd,
                    pinned.addressOf(offset + totalRead),
                    (length - totalRead).convert(),
                    (position + totalRead).convert()
                ).toInt()
                if (n < 0) {
                    val cause = strerror(errno)?.toKString() ?: "errno=$errno"
                    error("pread failed at offset ${position + totalRead}: $cause")
                }
                if (n == 0) break // EOF
                totalRead += n
            }
            totalRead
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        platform.posix.close(fd)
    }

    public companion object {
        /**
         * Open [path] for read-only random access. Returns `null` if the
         * file cannot be opened or stat'd — matches [JvmRandomAccessSource]
         * behaviour, letting consumers fall back to the legacy reader.
         */
        public fun open(path: String): PosixPreadRandomAccessSource? = memScoped {
            val fd = platform.posix.open(path, O_RDONLY)
            if (fd < 0) return@memScoped null
            val st = alloc<stat>()
            if (fstat(fd, st.ptr) != 0) {
                platform.posix.close(fd)
                return@memScoped null
            }
            PosixPreadRandomAccessSource(fd, st.st_size.toLong(), path)
        }
    }
}
