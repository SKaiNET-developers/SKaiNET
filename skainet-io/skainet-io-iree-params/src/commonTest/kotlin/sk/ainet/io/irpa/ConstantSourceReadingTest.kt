package sk.ainet.io.irpa

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.compile.hlo.ConstantSource

class ConstantSourceReadingTest {

    private fun leBytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 4)
        for ((i, v) in values.withIndex()) {
            val bits = v.toRawBits()
            for (b in 0 until 4) out[i * 4 + b] = (bits ushr (b * 8)).toByte()
        }
        return out
    }

    @Test
    fun bytesAreReadFromTheSourcesOffset() {
        val source = ConstantSource.Bytes(ByteArray(10) { it.toByte() }, offset = 2, sizeInBytes = 6)
        assertContentEquals(byteArrayOf(2, 3, 4, 5, 6, 7), source.readBytes(0, 6))
        assertContentEquals(byteArrayOf(5, 6), source.readBytes(3, 2))
    }

    @Test
    fun floatsAreReadAsLittleEndianBytesAtAnyByteOffset() {
        val values = floatArrayOf(1.5f, -2.25f, 3.0e-5f, Float.MAX_VALUE)
        val expected = leBytes(values)
        val source = ConstantSource.Floats(values)
        assertContentEquals(expected, source.readBytes(0, 16))
        assertContentEquals(expected.copyOfRange(3, 11), source.readBytes(3, 8))
    }

    @Test
    fun writeToGivesTheSameBytesAsReadBytes() {
        val floats = ConstantSource.Floats(FloatArray(1000) { it * 0.5f })
        val bytes = ConstantSource.Bytes(ByteArray(64) { (it * 3).toByte() }, offset = 4, sizeInBytes = 40)
        for (source in listOf(floats, bytes)) {
            val sink = Buffer()
            source.writeTo(sink)
            assertContentEquals(source.readBytes(0, source.sizeInBytes.toInt()), sink.readByteArray())
        }
    }

    @Test
    fun aRangeOutsideTheSourceIsRejected() {
        val source = ConstantSource.Floats(FloatArray(2))
        assertFailsWith<IllegalArgumentException> { source.readBytes(4, 8) }
        assertFailsWith<IllegalArgumentException> { source.readBytes(-1, 2) }
    }
}
