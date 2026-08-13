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
 * `androidNativeArm32`'s [RandomAccessSource], backed by POSIX `pread(2)`.
 *
 * Logic-identical to [PosixPreadRandomAccessSource] (native64Main) — `.convert()`
 * already targets whatever width `pread`'s generated signature expects — but kept
 * as a separate file/source set rather than shared, because androidNativeArm32's
 * `ssize_t`/`off_t` are 32-bit where every other native target is 64-bit: putting
 * both in one shared native source set fails Kotlin/Native's metadata compile
 * ("numbers with different bit widths"), see native64Main's own doc comment and
 * the `native64Targets` split in skainet-io-core's build.gradle.kts.
 *
 * 32-bit `off_t` without `_FILE_OFFSET_BITS=64` caps addressable file size at
 * 2 GiB — acceptable for this target's real workloads (GGUF/safetensors reads
 * on-device); large-file (100+ GB) loading stays a native64/JVM concern.
 */
@OptIn(ExperimentalForeignApi::class)
public class Posix32PreadRandomAccessSource private constructor(
    private val fd: Int,
    override val size: Long,
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
                    (position + totalRead).convert(),
                )
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
         * Open [path] for read-only random access. Returns `null` if the file
         * cannot be opened or stat'd, or exceeds the 32-bit `off_t` 2 GiB cap —
         * matching [PosixPreadRandomAccessSource]'s contract so callers can fall
         * back to the legacy sequential reader.
         */
        public fun open(path: String): Posix32PreadRandomAccessSource? = memScoped {
            val fd = platform.posix.open(path, O_RDONLY)
            if (fd < 0) return@memScoped null
            val st = alloc<stat>()
            if (fstat(fd, st.ptr) != 0) {
                platform.posix.close(fd)
                return@memScoped null
            }
            val size = st.st_size
            if (size < 0 || size > Int.MAX_VALUE.toLong()) {
                // Negative == 32-bit off_t overflow; >2GiB is unreachable on a
                // true 32-bit off_t anyway, but guard explicitly for clarity.
                platform.posix.close(fd)
                return@memScoped null
            }
            Posix32PreadRandomAccessSource(fd, size)
        }
    }
}
