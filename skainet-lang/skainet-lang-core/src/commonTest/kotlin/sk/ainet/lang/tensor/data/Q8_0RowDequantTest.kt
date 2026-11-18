package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** A packed Q8_0 table serves rows on demand, decoded like `toFloatArray()`. */
class Q8_0RowDequantTest {

    /** Packed bytes for [rows] x [cols]: every block gets its own scale, codes vary per element. */
    private fun table(rows: Int, cols: Int): Q8_0BlockTensorData {
        val blocks = (rows * cols + Q8_0TensorData.BLOCK_SIZE - 1) / Q8_0TensorData.BLOCK_SIZE
        val bytes = ByteArray(blocks * Q8_0TensorData.BYTES_PER_BLOCK)
        // f16 bit patterns for 0.5, 1.0, 2.0 (little-endian), cycled per block.
        val scales = intArrayOf(0x3800, 0x3C00, 0x4000)
        for (b in 0 until blocks) {
            val offset = b * Q8_0TensorData.BYTES_PER_BLOCK
            val scale = scales[b % scales.size]
            bytes[offset] = (scale and 0xFF).toByte()
            bytes[offset + 1] = (scale shr 8).toByte()
            for (i in 0 until Q8_0TensorData.BLOCK_SIZE) {
                bytes[offset + 2 + i] = ((b * 7 + i * 3) % 251 - 125).toByte()
            }
        }
        return Q8_0BlockTensorData(Shape(rows, cols), bytes)
    }

    private fun assertRowsMatchFullDequant(rows: Int, cols: Int) {
        val data = table(rows, cols)
        val all = data.toFloatArray()
        for (r in 0 until rows) {
            assertContentEquals(all.copyOfRange(r * cols, (r + 1) * cols), data.dequantRow(r), "row $r")
        }
    }

    @Test
    fun `q8_0 table is a RowDequantSource`() {
        assertTrue(table(2, 32) is RowDequantSource)
    }

    @Test
    fun `rows aligned to blocks match full dequantization`() = assertRowsMatchFullDequant(rows = 5, cols = 64)

    @Test
    fun `rows that straddle blocks match full dequantization`() = assertRowsMatchFullDequant(rows = 7, cols = 20)

    @Test
    fun `rows wider than a block with a ragged tail match full dequantization`() =
        assertRowsMatchFullDequant(rows = 3, cols = 50)

    @Test
    fun `out of range rows and non tables and relayouted weights are refused`() {
        val data = table(4, 32)
        assertFailsWith<IllegalArgumentException> { data.dequantRow(4) }
        assertFailsWith<IllegalArgumentException> { data.dequantRow(-1) }
        val vector = Q8_0BlockTensorData(Shape(32), ByteArray(Q8_0TensorData.BYTES_PER_BLOCK))
        assertFailsWith<IllegalArgumentException> { vector.dequantRow(0) }
        val relayouted = Q8_0BlockTensorData(
            Shape(4, 32), data.packedData, BlockOrder.INPUT_BLOCK_MAJOR, blockAxis = 0,
        )
        assertFailsWith<IllegalArgumentException> { relayouted.dequantRow(0) }
    }
}
