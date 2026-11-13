package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * The `[in, out]` label over a relayouted weight says what it is (SKEEP-003, #1507): feed-order
 * bytes whose blocks run along axis 0. Its view and its `toFloatArray()` then decode the transpose
 * of the weight, where the unmarked label decoded the wrong blocks.
 */
class PackedBlockAxisTest {
    private val rows = 3            // out
    private val blocksPerRow = 2
    private val cols = blocksPerRow * 32   // in

    /** Canonical Q8_0 bytes: fp16 scale 1.0, codes that differ per element. */
    private fun canonicalBytes(): ByteArray {
        val bytes = ByteArray(rows * blocksPerRow * 34)
        var seed = 7
        for (b in 0 until rows * blocksPerRow) {
            val base = b * 34
            bytes[base] = 0x00; bytes[base + 1] = 0x3C
            for (i in 0 until 32) {
                seed = seed * 1103515245 + 12345
                bytes[base + 2 + i] = ((seed ushr 16) % 23 - 11).toByte()
            }
        }
        return bytes
    }

    private fun feedOrder(canonical: ByteArray): ByteArray {
        val out = ByteArray(canonical.size)
        for (o in 0 until rows) for (b in 0 until blocksPerRow) {
            canonical.copyInto(out, (b * rows + o) * 34, (o * blocksPerRow + b) * 34, (o * blocksPerRow + b + 1) * 34)
        }
        return out
    }

    @Test
    fun theLabelDecodesTheTransposeOfTheWeight() {
        val canonical = Q8_0BlockTensorData(Shape(rows, cols), canonicalBytes())
        val weight = canonical.toFloatArray()                     // [out, in], row-major
        val label = Q8_0BlockTensorData(
            Shape(cols, rows), feedOrder(canonicalBytes()), BlockOrder.INPUT_BLOCK_MAJOR, blockAxis = 0,
        )

        val transposed = FloatArray(cols * rows)
        for (o in 0 until rows) for (i in 0 until cols) transposed[i * rows + o] = weight[o * cols + i]

        assertContentEquals(transposed, (label as PackedBlockStorage).toFloatArray())
        val view = label.view
        assertEquals(Shape(cols, rows), view.shape)
        assertEquals(0, view.layout.blockAxis)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, view.layout.blockOrder)
        for (i in 0 until cols) for (o in 0 until rows) {
            assertEquals(weight[o * cols + i], view.get(i, o), "element ($i, $o)")
        }
        // Transposed back, it is the feed-order weight the packed kernels are keyed on.
        val back = view.transpose()
        assertEquals(Shape(rows, cols), back.shape)
        assertEquals(1, back.layout.blockAxis)
        assertContentEquals(intArrayOf(1, rows), back.layout.strides)
    }

    @Test
    fun theDefaultBlockAxisIsTheLastOne() {
        val canonical = Q8_0BlockTensorData(Shape(rows, cols), canonicalBytes())
        assertEquals(1, canonical.blockAxis)
        assertEquals(1, canonical.view.layout.blockAxis)
    }

    @Test
    fun theFactoryMapsEveryBlockEncodingToItsClass() {
        val expected = mapOf(
            TensorEncoding.Q4_0 to (18 to 32), TensorEncoding.Q5_0 to (22 to 32), TensorEncoding.Q5_1 to (24 to 32),
            TensorEncoding.Q8_0 to (34 to 32), TensorEncoding.Q4_K to (144 to 256), TensorEncoding.Q5_K to (176 to 256),
            TensorEncoding.Q6_K to (210 to 256),
        )
        for ((encoding, geometry) in expected) {
            val (bytesPerBlock, blockSize) = geometry
            val data = assertNotNull(
                packedBlockTensorData(encoding, Shape(blockSize, 2), ByteArray(2 * bytesPerBlock), BlockOrder.INPUT_BLOCK_MAJOR, blockAxis = 0),
                "${encoding.name} has a block class",
            )
            val packed = data as PackedBlockStorage
            assertEquals(encoding, packed.encoding)
            assertEquals(0, packed.blockAxis)
            assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, packed.blockOrder)
        }
        assertNull(packedBlockTensorData(TensorEncoding.Dense(4), Shape(2, 2), ByteArray(16)))
    }
}
