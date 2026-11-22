package sk.ainet.io.irpa

import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.compile.hlo.ConstantSource

class ConstantSourceFileRangeReadingTest {

    @Test
    fun aFileRangeIsReadInPartsAndWrittenWhole() {
        val file = File.createTempFile("constant-source", ".bin")
        try {
            val content = ByteArray(4096) { (it * 7).toByte() }
            file.writeBytes(content)
            val source = ConstantSource.FileRange(file.path, fileOffset = 100, sizeInBytes = 1000)

            assertContentEquals(content.copyOfRange(100, 1100), source.readBytes(0, 1000))
            assertContentEquals(content.copyOfRange(350, 414), source.readBytes(250, 64))

            val sink = Buffer()
            source.writeTo(sink)
            assertContentEquals(content.copyOfRange(100, 1100), sink.readByteArray())
        } finally {
            file.delete()
        }
    }
}
