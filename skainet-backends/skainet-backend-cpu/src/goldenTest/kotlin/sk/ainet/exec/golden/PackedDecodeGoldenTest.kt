package sk.ainet.exec.golden

import sk.ainet.exec.golden.GoldenSupport.Packed
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import kotlin.test.Test

/**
 * SKEEP-003 golden gate, decode half: the block decoders of every packed `TensorData` must produce
 * bit-identical floats for the same bytes. Guards the TensorData → TensorView façade migration
 * (M1) and every later refactor of the packed storage types.
 */
class PackedDecodeGoldenTest {

    private companion object {
        const val ROWS = 4
        const val BLOCKS_PER_ROW = 3
        const val SEED = 0x5EED_0001L
    }

    private fun decodeAll(storage: PackedBlockStorage, elementCount: Int, blockSize: Int): FloatArray {
        val out = FloatArray(elementCount)
        val blocks = (elementCount + blockSize - 1) / blockSize
        val tmp = FloatArray(blockSize)
        for (b in 0 until blocks) {
            storage.dequantizeBlock(b, tmp, 0)
            val n = minOf(blockSize, elementCount - b * blockSize)
            tmp.copyInto(out, b * blockSize, 0, n)
        }
        return out
    }

    private fun golden(p: Packed, build: (Shape, ByteArray) -> PackedBlockStorage) {
        val blocks = GoldenSupport.weightBlocks(p, ROWS, BLOCKS_PER_ROW, SEED)
        val shape = Shape(ROWS, BLOCKS_PER_ROW * p.blockSize)
        val storage = build(shape, GoldenSupport.rowMajor(blocks))
        val decoded = decodeAll(storage, shape.volume, p.blockSize)
        GoldenSupport.check("decode/${p.name}", GoldenSupport.digest(decoded))
    }

    @Test fun q4_0() = golden(Packed.Q4_0) { s, b -> Q4_0BlockTensorData(s, b) }
    @Test fun q5_0() = golden(Packed.Q5_0) { s, b -> Q5_0BlockTensorData(s, b) }
    @Test fun q5_1() = golden(Packed.Q5_1) { s, b -> Q5_1BlockTensorData(s, b) }
    @Test fun q8_0() = golden(Packed.Q8_0) { s, b -> Q8_0BlockTensorData(s, b) }
    @Test fun q4_K() = golden(Packed.Q4_K) { s, b -> Q4_KBlockTensorData(s, b) }
    @Test fun q5_K() = golden(Packed.Q5_K) { s, b -> Q5_KBlockTensorData(s, b) }
    @Test fun q6_K() = golden(Packed.Q6_K) { s, b -> Q6_KBlockTensorData(s, b) }

    @Test
    fun ternaryFromValues() {
        val rng = GoldenSupport.Rng(SEED + 7)
        val n = 4 * 96
        val values = ByteArray(n) { (rng.nextByte().toInt() % 3).toByte() } // -2..2 → clamp to {-1,0,1}
        for (i in values.indices) values[i] = values[i].toInt().coerceIn(-1, 1).toByte()
        val t = Ternary2BitTensorData.fromTernaryValues(Shape(4, 96), values, scale = 0.8125f)
        val decoded = decodeAll(t, n, t.blockSize)
        GoldenSupport.check("decode/TERNARY_values", GoldenSupport.digest(decoded))
        GoldenSupport.check("packed/TERNARY_values", GoldenSupport.digest(t.packedData))
    }

    @Test
    fun ternaryFromTQ2_0Block() {
        val rng = GoldenSupport.Rng(SEED + 11)
        val block = ByteArray(66) { rng.nextByte() }
        GoldenSupport.le16(block, 64, GoldenSupport.half(0.03125f))
        val t = Ternary2BitTensorData.fromTQ2_0Block(block, Shape(256))
        val decoded = decodeAll(t, 256, t.blockSize)
        GoldenSupport.check("decode/TERNARY_tq2_0", GoldenSupport.digest(decoded))
    }
}
