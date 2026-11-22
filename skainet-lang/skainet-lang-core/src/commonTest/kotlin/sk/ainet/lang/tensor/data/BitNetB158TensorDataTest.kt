package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * #1140: the first packed ternary `TensorData`. Its decode must be pinned to [TernaryCodec] —
 * the same reference the kernels and the GGUF loader are defined against — so all three readers
 * agree about the same bytes.
 */
class BitNetB158TensorDataTest {

    private fun ternaryValues(count: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(count) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1) * 0.5f
        }
    }

    @Test
    fun roundTripsThroughTheCodec() {
        val n = 6; val k = 32
        val values = ternaryValues(n * k, seed = 5)
        val data = BitNetB158TensorData.fromFloats(Shape(n, k), values)

        assertEquals(TernaryCodec.bitNetScale(data.packedData, n * k), data.scale)
        assertContentEquals(
            TernaryCodec.decodeBitNet(data.packedData, n * k),
            data.toFloatArray(),
            "toFloatArray must equal the codec's decode of the same bytes",
        )
    }

    @Test
    fun getReturnsTheDecodedValueAndSetIsRefused() {
        // 0.5 everywhere it is not zero, so the per-tensor scale is 0.5 and a decoded value differs
        // from its code: `get` must give the value (SKEEP-003 rule 3), not the code.
        val values = floatArrayOf(0.5f, -0.5f, 0f, 0.5f, -0.5f, 0f, 0f, 0.5f)
        val data = BitNetB158TensorData.fromFloats(Shape(8), values)
        val decoded = data.toFloatArray()
        for (i in values.indices) assertEquals(decoded[i], data.get(i), "element $i")
        assertEquals(data.scale, data.get(0))
        assertEquals(-data.scale, data.get(1))
        assertEquals(0f, data.get(2))
        assertFailsWith<UnsupportedOperationException> { data.set(2, value = -1f) }
    }

    @Test
    fun viewCarriesTheExactDispatchFormat() {
        val data = BitNetB158TensorData.fromFloats(Shape(4, 8), ternaryValues(32, seed = 9))
        val view = data.packedView
        assertEquals(TensorEncoding.BITNET_B1_58, view.format.encoding)
        assertEquals(1, data.blockCount, "per-tensor encoding: one block")
        assertEquals(32, data.blockSize)
        // the view decodes with the scale applied, exactly like the codec
        val decoded = TernaryCodec.decodeBitNet(data.packedData, 32)
        assertEquals(decoded[9], view.get(1, 1))
    }

    @Test
    fun rejectsBuffersWithoutTheScaleTrailer() {
        assertFailsWith<IllegalArgumentException> {
            BitNetB158TensorData(Shape(8), ByteArray(2)) // payload only, no trailer
        }
    }
}
