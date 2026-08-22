@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StorageToTensorDataTest {

    @Test
    fun roundTripDenseFloat32() {
        // TensorData → TensorStorage → TensorData
        val original = DenseFloatArrayTensorData<FP32>(Shape(3), floatArrayOf(1f, 2f, 3f))
        val storage = TensorStorageFactory.fromTensorData(original)
        val restored = TensorStorageFactory.toTensorData<FP32, Float>(storage)

        assertTrue(restored is FloatArrayTensorData<*>)
        val floats = (restored as FloatArrayTensorData<*>).buffer
        assertEquals(3, floats.size)
        assertEquals(1f, floats[0])
        assertEquals(2f, floats[1])
        assertEquals(3f, floats[2])
    }

    @Test
    fun roundTripDenseInt32() {
        val original = DenseIntArrayTensorData<Int32>(Shape(4), intArrayOf(10, 20, 30, 40))
        val storage = TensorStorageFactory.fromTensorData(original)
        val restored = TensorStorageFactory.toTensorData<Int32, Int>(storage)

        assertTrue(restored is IntArrayTensorData<*>)
        val ints = (restored as IntArrayTensorData<*>).buffer
        assertEquals(4, ints.size)
        assertEquals(10, ints[0])
        assertEquals(40, ints[3])
    }

    @Test
    fun roundTripQ4K() {
        val rawBytes = ByteArray(144) // 1 Q4_K block
        rawBytes[10] = 42 // put something non-zero to verify identity
        val original = Q4_KBlockTensorData.fromRawBytes(Shape(256), rawBytes)
        val storage = TensorStorageFactory.fromTensorData(original)

        assertEquals(TensorEncoding.Q4_K, storage.encoding)

        val restored = TensorStorageFactory.toTensorData<DType, Byte>(storage)
        assertTrue(restored is Q4_KTensorData)
        assertEquals(256, restored.shape.volume)
        assertEquals(42, (restored as Q4_KTensorData).packedData[10])
    }

    @Test
    fun roundTripQ80() {
        // Build a Q8_0 block: scale=1.0 (f16 0x3C00) + 32 code bytes
        val rawBytes = ByteArray(34)
        rawBytes[0] = 0x00
        rawBytes[1] = 0x3C
        for (i in 0 until 32) rawBytes[2 + i] = (i + 1).toByte()

        val original = Q8_0BlockTensorData.fromRawBytes(Shape(32), rawBytes)
        val storage = TensorStorageFactory.fromTensorData(original)

        assertEquals(TensorEncoding.Q8_0, storage.encoding)

        val restored = TensorStorageFactory.toTensorData<DType, Byte>(storage)
        assertTrue(restored is Q8_0TensorData)
        val q80 = restored as Q8_0TensorData
        assertEquals(32, q80.shape.volume)
        // Verify codes are intact
        assertEquals(1.toByte(), q80.getCode(0, 0))
        assertEquals(32.toByte(), q80.getCode(0, 31))
    }

    @Test
    fun toTensorDataFromBorrowedFloat32() {
        // Create storage from raw bytes directly
        val floatBytes = ByteArray(12) // 3 floats
        // 1.0f = 0x3F800000 little-endian
        floatBytes[0] = 0x00; floatBytes[1] = 0x00; floatBytes[2] = 0x80.toByte(); floatBytes[3] = 0x3F
        // 2.0f = 0x40000000
        floatBytes[4] = 0x00; floatBytes[5] = 0x00; floatBytes[6] = 0x00; floatBytes[7] = 0x40
        // 3.0f = 0x40400000
        floatBytes[8] = 0x00; floatBytes[9] = 0x00; floatBytes[10] = 0x40; floatBytes[11] = 0x40

        val storage = TensorStorageFactory.fromRawBytes(
            shape = Shape(3),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            data = floatBytes
        )

        val td = TensorStorageFactory.toTensorData<FP32, Float>(storage)
        assertTrue(td is FloatArrayTensorData<*>)
        assertEquals(1f, (td as FloatArrayTensorData<*>).buffer[0])
        assertEquals(2f, td.buffer[1])
        assertEquals(3f, td.buffer[2])
    }
}
