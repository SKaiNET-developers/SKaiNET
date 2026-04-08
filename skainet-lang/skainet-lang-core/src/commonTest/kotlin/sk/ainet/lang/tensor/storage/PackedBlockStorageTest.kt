package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PackedBlockStorageTest {

    @Test
    fun q4kImplementsPackedBlockStorage() {
        val data = ByteArray(144) // 1 Q4_K block
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), data)

        assertTrue(td is PackedBlockStorage)
        val packed = td as PackedBlockStorage
        assertEquals(TensorEncoding.Q4_K, packed.encoding)
        assertEquals(256, packed.blockSize)
        assertEquals(1, packed.blockCount)
        assertEquals(144L, packed.physicalBytes)
        assertEquals(256L, packed.elementCount)
    }

    @Test
    fun q80ImplementsPackedBlockStorage() {
        val data = ByteArray(34) // 1 Q8_0 block
        val td = Q8_0BlockTensorData.fromRawBytes(Shape(32), data)

        assertTrue(td is PackedBlockStorage)
        val packed = td as PackedBlockStorage
        assertEquals(TensorEncoding.Q8_0, packed.encoding)
        assertEquals(32, packed.blockSize)
        assertEquals(1, packed.blockCount)
        assertEquals(34L, packed.physicalBytes)
        assertEquals(32L, packed.elementCount)
    }

    @Test
    fun q80DequantizeBlockProducesCorrectOutput() {
        // Create a Q8_0 block: 2 bytes scale (f16 for 1.0) + 32 bytes codes
        val data = ByteArray(34)
        // Scale = 1.0 in f16: sign=0, exp=15, mant=0 → 0x3C00 (little-endian: 0x00, 0x3C)
        data[0] = 0x00.toByte()
        data[1] = 0x3C.toByte()
        // Codes: 1, 2, 3, ... 32
        for (i in 0 until 32) {
            data[2 + i] = (i + 1).toByte()
        }

        val td = Q8_0BlockTensorData.fromRawBytes(Shape(32), data)
        val packed = td as PackedBlockStorage
        val output = FloatArray(32)
        packed.dequantizeBlock(0, output)

        // output[i] = code[i] * scale = (i+1) * 1.0
        for (i in 0 until 32) {
            assertEquals((i + 1).toFloat(), output[i], "Element $i")
        }
    }

    @Test
    fun q80ToFloatArrayDequantizesAll() {
        val data = ByteArray(34)
        data[0] = 0x00.toByte() // scale = 1.0 f16
        data[1] = 0x3C.toByte()
        for (i in 0 until 32) {
            data[2 + i] = (i + 1).toByte()
        }

        val td = Q8_0BlockTensorData.fromRawBytes(Shape(32), data)
        val packed = td as PackedBlockStorage
        val floats = packed.toFloatArray()

        assertEquals(32, floats.size)
        assertEquals(1.0f, floats[0])
        assertEquals(32.0f, floats[31])
    }

    @Test
    fun packedBlockStorageToTensorStorage() {
        val data = ByteArray(144)
        val td = Q4_KBlockTensorData.fromRawBytes(Shape(256), data)
        val packed = td as PackedBlockStorage
        val storage = packed.toTensorStorage()

        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(TensorEncoding.Q4_K, storage.encoding)
        assertEquals(Ownership.BORROWED, storage.ownership)
        assertEquals(144L, storage.physicalBytes)
        assertEquals(1024L, storage.logicalBytes) // 256 * 4
    }
}
