package sk.ainet.io.wav

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

public actual fun readWav(path: String): WavData {
    val file = File(path)
    require(file.exists()) { "File not found: $path" }
    val bytes = file.readBytes()
    require(bytes.size >= 44) { "File too small to be a valid WAV: $path" }

    fun ascii(start: Int, end: Int): String =
        bytes.copyOfRange(start, end).toString(Charsets.US_ASCII)

    require(ascii(0, 4) == "RIFF") { "Missing RIFF header" }
    require(ascii(8, 12) == "WAVE") { "Missing WAVE header" }

    var cursor = 12
    var sampleRate = 0
    var bitsPerSample = 0
    var channels = 0
    var dataOffset = -1
    var dataSize = -1

    fun leUInt(offset: Int): Long = ByteBuffer.wrap(bytes, offset, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
    fun leUShort(offset: Int): Int = ByteBuffer.wrap(bytes, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
    fun leShort(offset: Int): Short = ByteBuffer.wrap(bytes, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short

    while (cursor + 8 <= bytes.size) {
        val chunkId = ascii(cursor, cursor + 4)
        val chunkSize = leUInt(cursor + 4).toInt()
        val next = cursor + 8 + chunkSize
        when (chunkId) {
            "fmt " -> {
                val audioFormat = leUShort(cursor + 8)
                require(audioFormat == 1) { "Only PCM WAV is supported (found $audioFormat)" }
                channels = leUShort(cursor + 10)
                sampleRate = leUInt(cursor + 12).toInt()
                bitsPerSample = leUShort(cursor + 22)
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
            val sample = leShort(src).toFloat() / Short.MAX_VALUE
            acc += sample
            src += bytesPerSample
            c++
        }
        out[i] = acc / channels
        i++
    }

    return WavData(out, sampleRate)
}
