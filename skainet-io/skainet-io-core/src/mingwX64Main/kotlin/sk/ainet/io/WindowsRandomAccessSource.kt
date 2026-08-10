package sk.ainet.io

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.windows.CloseHandle
import platform.windows.CreateFileW
import platform.windows.DWORDVar
import platform.windows.ERROR_HANDLE_EOF
import platform.windows.FILE_ATTRIBUTE_NORMAL
import platform.windows.FILE_SHARE_READ
import platform.windows.GENERIC_READ
import platform.windows.GetFileSizeEx
import platform.windows.GetLastError
import platform.windows.HANDLE
import platform.windows.INVALID_HANDLE_VALUE
import platform.windows.LARGE_INTEGER
import platform.windows.OPEN_EXISTING
import platform.windows.OVERLAPPED
import platform.windows.ReadFile

/**
 * Windows [RandomAccessSource] backed by `ReadFile` with an `OVERLAPPED` offset.
 *
 * POSIX `pread(2)` does not exist on mingw; passing an `OVERLAPPED` structure with an
 * explicit 64-bit offset to `ReadFile` gives the same positional semantics — every call
 * names its own offset, so concurrent reads from different positions are safe without
 * locking, and files > 2 GB work (offset is split into `Offset`/`OffsetHigh`).
 *
 * This is deliberately a separate leaf implementation: `mingwX64Main` must stay out of
 * this module's `native64Main` source set (that set is POSIX-`pread`-shaped and LP64;
 * mingw is LLP64). [close] is single-shot. See #911.
 */
@OptIn(ExperimentalForeignApi::class)
public class WindowsRandomAccessSource private constructor(
    private val handle: HANDLE,
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
                val chunk = memScoped {
                    val pos = position + totalRead
                    val overlapped = alloc<OVERLAPPED>()
                    overlapped.Offset = (pos and 0xFFFF_FFFFL).toUInt()
                    overlapped.OffsetHigh = (pos ushr 32).toUInt()
                    val read = alloc<DWORDVar>()
                    val ok = ReadFile(
                        handle,
                        pinned.addressOf(offset + totalRead),
                        (length - totalRead).convert(),
                        read.ptr,
                        overlapped.ptr,
                    )
                    if (ok == 0) {
                        val err = GetLastError()
                        if (err == ERROR_HANDLE_EOF.toUInt()) return@memScoped 0
                        error("ReadFile failed at offset $pos: Win32 error $err")
                    }
                    read.value.toInt()
                }
                if (chunk == 0) break // EOF
                totalRead += chunk
            }
            totalRead
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        CloseHandle(handle)
    }

    public companion object {
        /**
         * Open [path] for read-only random access. Returns `null` if the file cannot be
         * opened or sized — matching the JVM/POSIX implementations, so consumers fall
         * back to the legacy sequential reader.
         */
        public fun open(path: String): WindowsRandomAccessSource? {
            val handle = CreateFileW(
                path,
                GENERIC_READ.convert(),
                FILE_SHARE_READ.convert(),
                null,
                OPEN_EXISTING.convert(),
                FILE_ATTRIBUTE_NORMAL.convert(),
                null,
            )
            if (handle == null || handle == INVALID_HANDLE_VALUE) return null
            return memScoped {
                val sizeVar = alloc<LARGE_INTEGER>()
                if (GetFileSizeEx(handle, sizeVar.ptr) == 0) {
                    CloseHandle(handle)
                    null
                } else {
                    WindowsRandomAccessSource(handle, sizeVar.QuadPart)
                }
            }
        }
    }
}
