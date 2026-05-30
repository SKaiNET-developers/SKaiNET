package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class Q4_0TensorDataTest {

    /** Pack 32 unsigned 4-bit codes (0..15) into the canonical split layout. */
    private fun packCodes(codes: IntArray): ByteArray {
        require(codes.size == 32)
        val out = ByteArray(16)
        for (j in 0 until 16) {
            out[j] = ((codes[j] and 0x0F) or ((codes[j + 16] and 0x0F) shl 4)).toByte()
        }
        return out
    }

    private fun block(scaleLo: Int, scaleHi: Int, codes: IntArray): ByteArray =
        byteArrayOf(scaleLo.toByte(), scaleHi.toByte()) + packCodes(codes)

    @Test
    fun `constants are correct`() {
        assertEquals(32, Q4_0TensorData.BLOCK_SIZE)
        assertEquals(18, Q4_0TensorData.BYTES_PER_BLOCK)
    }

    @Test
    fun `reads scale from block`() {
        // scale = 1.0 (f16 0x3C00 little-endian)
        val data = block(0x00, 0x3C, IntArray(32) { 8 })
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(32), data)
        assertEquals(1.0f, tensor.getBlockScale(0), 0.001f)
    }

    @Test
    fun `split layout decodes low nibbles to first half and high nibbles to second half`() {
        // codes[j]=j%16 → low nibble j∈0..15 ; codes[j+16]=15-(j%16) → high nibble
        val codes = IntArray(32) { i -> if (i < 16) i else 15 - (i - 16) }
        val data = block(0x00, 0x3C, codes) // scale 1.0
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(32), data)
        for (i in 0 until 32) {
            assertEquals(codes[i].toByte(), tensor.getCode(0, i), "code mismatch at $i")
        }
    }

    @Test
    fun `toFloatArray applies minus-eight bias and scale`() {
        // scale = 0.5 (f16 0x3800). codes: elem0=10 → (10-8)*0.5=1.0 ; elem16=6 → (6-8)*0.5=-1.0
        val codes = IntArray(32) { 8 }
        codes[0] = 10   // low nibble of byte 0  → element 0
        codes[16] = 6   // high nibble of byte 0 → element 16
        val data = block(0x00, 0x38, codes)
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(32), data)
        val floats = tensor.toFloatArray()
        assertEquals(1.0f, floats[0], 0.01f)
        assertEquals(-1.0f, floats[16], 0.01f)
        assertEquals(0.0f, floats[1], 0.01f) // code 8 → (8-8)*scale = 0
    }

    @Test
    fun `matches canonical ggml dequant for a known block`() {
        // Mirror DequantOps.dequantQ4_0FromBytes: out[j]=(lo-8)*d, out[j+16]=(hi-8)*d.
        val codes = IntArray(32) { i -> (i * 7 + 3) and 0x0F } // arbitrary 0..15 pattern
        val data = block(0x00, 0x3C, codes) // scale 1.0
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(32), data)
        val floats = tensor.toFloatArray()
        for (i in 0 until 32) {
            assertEquals((codes[i] - 8).toFloat(), floats[i], 0.001f, "dequant mismatch at $i")
        }
    }

    @Test
    fun `set round-trips through nibble packing`() {
        val data = block(0x00, 0x3C, IntArray(32) { 8 })
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(32), data)
        tensor[3] = 5      // low nibble of byte 3
        tensor[19] = 12    // high nibble of byte 3 (19-16=3)
        assertEquals(5.toByte(), tensor[3])
        assertEquals(12.toByte(), tensor[19])
    }

    @Test
    fun `handles multiple blocks and 2D shape`() {
        val b0 = block(0x00, 0x3C, IntArray(32) { 8 })  // scale 1.0
        val b1 = block(0x00, 0x40, IntArray(32) { 9 })  // scale 2.0, code 9
        val tensor = Q4_0BlockTensorData.fromRawBytes(Shape(8, 8), b0 + b1)
        assertEquals(2, tensor.blockCount)
        assertContentEquals(intArrayOf(8, 8), tensor.shape.dimensions)
        assertEquals(1.0f, tensor.getBlockScale(0), 0.001f)
        assertEquals(2.0f, tensor.getBlockScale(1), 0.001f)
        assertEquals(9.toByte(), tensor.getCode(1, 0))
    }
}
