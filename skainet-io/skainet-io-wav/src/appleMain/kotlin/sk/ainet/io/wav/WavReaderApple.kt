@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package sk.ainet.io.wav

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import platform.posix.SEEK_END
import platform.posix.SEEK_SET
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fread
import platform.posix.fseek
import platform.posix.ftell

public actual fun readWav(path: String): WavData {
    val bytes = readFile(path)
    require(bytes.size >= 44) { "File too small to be a valid WAV: $path" }
    require(bytes.ascii(0, 4) == "RIFF") { "Missing RIFF header" }
    require(bytes.ascii(8, 12) == "WAVE") { "Missing WAVE header" }

    var cursor = 12
    var sampleRate = 0
    var bitsPerSample = 0
    var channels = 0
    var dataOffset = -1
    var dataSize = -1

    while (cursor + 8 <= bytes.size) {
        val chunkId = bytes.ascii(cursor, cursor + 4)
        val chunkSize = bytes.leUInt(cursor + 4).toInt()
        val next = cursor + 8 + chunkSize
        when (chunkId) {
            "fmt " -> {
                val audioFormat = bytes.leUShort(cursor + 8)
                require(audioFormat == 1) { "Only PCM WAV is supported (found format $audioFormat)" }
                channels = bytes.leUShort(cursor + 10)
                sampleRate = bytes.leUInt(cursor + 12).toInt()
                bitsPerSample = bytes.leUShort(cursor + 22)
            }

            "data" -> {
                dataOffset = cursor + 8
                dataSize = chunkSize
                break
            }
        }
        cursor = next
    }

    require(sampleRate > 0 && bitsPerSample > 0 && channels > 0) { "Incomplete WAV fmt chunk" }
    require(bitsPerSample == 16) { "Only 16-bit PCM is supported (found $bitsPerSample)" }
    require(dataOffset >= 0 && dataSize > 0) { "Missing data chunk in WAV file" }
    require(dataOffset + dataSize <= bytes.size) { "Data chunk exceeds file size" }

    val bytesPerSample = bitsPerSample / 8
    val totalSamples = dataSize / (bytesPerSample * channels)
    val out = FloatArray(totalSamples)
    var src = dataOffset
    var i = 0
    while (i < totalSamples) {
        var acc = 0f
        var c = 0
        while (c < channels) {
            val sample = bytes.leShort(src).toFloat() / Short.MAX_VALUE
            acc += sample
            src += bytesPerSample
            c++
        }
        out[i] = acc / channels
        i++
    }

    return WavData(out, sampleRate)
}

private fun readFile(path: String): ByteArray {
    val file = fopen(path, "rb") ?: error("Unable to open file: $path")
    try {
        fseek(file, 0, SEEK_END)
        val size = ftell(file)
        fseek(file, 0, SEEK_SET)
        require(size > 0) { "Empty file: $path" }
        val buffer = ByteArray(size.toInt())
        buffer.usePinned { pinned ->
            val read = fread(pinned.addressOf(0), 1.convert(), size.convert(), file)
            require(read.toLong() == size) { "Read $read bytes, expected $size from $path" }
        }
        return buffer
    } finally {
        fclose(file)
    }
}

private fun ByteArray.leUInt(offset: Int): Long {
    return (this[offset].toLong() and 0xFF) or
        ((this[offset + 1].toLong() and 0xFF) shl 8) or
        ((this[offset + 2].toLong() and 0xFF) shl 16) or
        ((this[offset + 3].toLong() and 0xFF) shl 24)
}

private fun ByteArray.leUShort(offset: Int): Int {
    return ((this[offset].toInt() and 0xFF) or
        ((this[offset + 1].toInt() and 0xFF) shl 8))
}

private fun ByteArray.leShort(offset: Int): Short {
    val low = this[offset].toInt() and 0xFF
    val high = this[offset + 1].toInt() and 0xFF
    return ((high shl 8) or low).toShort()
}

private fun ByteArray.ascii(start: Int, end: Int): String {
    val len = end - start
    val builder = StringBuilder(len)
    var i = 0
    while (i < len) {
        builder.append(this[start + i].toInt().toChar())
        i++
    }
    return builder.toString()
}
