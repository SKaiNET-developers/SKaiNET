package sk.ainet.io.irpa

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * #1247: a [BufferHandle.Floats] entry must produce byte-identical archive
 * output to the same values serialized eagerly into [BufferHandle.Owned] —
 * the Floats handle only removes the up-front byte copy (and with it the
 * 2 GiB single-array ceiling), never changes the on-disk format.
 */
class IrpaFloatsParityTest {

    @Test
    fun floats_entry_is_byte_identical_to_owned_serialization() {
        val values = FloatArray(1027) { (it - 500) * 0.37f } // odd size: crosses chunk lane logic
        val bytes = ByteArray(values.size * 4)
        for (i in values.indices) {
            val bits = values[i].toRawBits()
            bytes[i * 4] = (bits and 0xff).toByte()
            bytes[i * 4 + 1] = (bits ushr 8 and 0xff).toByte()
            bytes[i * 4 + 2] = (bits ushr 16 and 0xff).toByte()
            bytes[i * 4 + 3] = (bits ushr 24 and 0xff).toByte()
        }

        fun ref(source: BufferHandle) = ExternalParameterRef(
            scope = "model",
            key = "w",
            encoding = TensorEncoding.Dense(bytesPerElement = 4),
            source = source
        )

        val ownedOut = Buffer().also { IrpaWriter().write(listOf(ref(BufferHandle.Owned(bytes))), it) }
        val floatsOut = Buffer().also { IrpaWriter().write(listOf(ref(BufferHandle.Floats(values))), it) }

        assertContentEquals(ownedOut.readByteArray(), floatsOut.readByteArray())
    }
}
