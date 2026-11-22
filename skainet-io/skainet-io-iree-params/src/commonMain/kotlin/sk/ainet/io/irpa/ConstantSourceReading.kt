package sk.ainet.io.irpa

import kotlinx.io.Buffer
import kotlinx.io.Sink
import kotlinx.io.readByteArray
import sk.ainet.compile.hlo.ConstantSource

/**
 * [length] bytes of this source, starting at byte [offset] of its [ConstantSource.sizeInBytes]
 * bytes, in the form a packager writes them: the bytes of a [ConstantSource.Bytes], the
 * little-endian FP32 bytes of a [ConstantSource.Floats], the file's bytes of a
 * [ConstantSource.FileRange].
 *
 * Read a large source in chunks: a [ConstantSource.Floats] above 2 GiB has no single byte array.
 * A file range is opened for each call, where the platform can read one (see [IrpaWriter]).
 */
public fun ConstantSource.readBytes(offset: Long, length: Int): ByteArray {
    require(offset >= 0 && length >= 0 && offset + length <= sizeInBytes) {
        "range [$offset, ${offset + length}) is outside the source's $sizeInBytes bytes"
    }
    if (length == 0) return ByteArray(0)
    return when (this) {
        is ConstantSource.Bytes -> {
            val start = this.offset + offset.toInt()
            data.copyOfRange(start, start + length)
        }
        is ConstantSource.Floats -> {
            val out = ByteArray(length)
            for (i in 0 until length) {
                val at = offset + i
                val bits = data[(at ushr 2).toInt()].toRawBits()
                out[i] = (bits ushr ((at and 3L).toInt() * 8)).toByte()
            }
            out
        }
        is ConstantSource.FileRange -> {
            val buffer = Buffer()
            writeFileRangeBytes(buffer, ConstantSource.FileRange(path, fileOffset + offset, length.toLong()))
            buffer.readByteArray()
        }
    }
}

/**
 * All [ConstantSource.sizeInBytes] bytes of this source into [sink], in the form a packager
 * writes them, without holding them in one array. The sink is not flushed or closed.
 */
public fun ConstantSource.writeTo(sink: Sink) {
    when (this) {
        is ConstantSource.Bytes -> sink.write(data, offset, offset + sizeInBytes.toInt())
        is ConstantSource.FileRange -> writeFileRangeBytes(sink, this)
        is ConstantSource.Floats -> {
            val chunk = 1 shl 20
            var position = 0L
            while (position < sizeInBytes) {
                val count = minOf(chunk.toLong(), sizeInBytes - position).toInt()
                sink.write(readBytes(position, count))
                position += count
            }
        }
    }
}
