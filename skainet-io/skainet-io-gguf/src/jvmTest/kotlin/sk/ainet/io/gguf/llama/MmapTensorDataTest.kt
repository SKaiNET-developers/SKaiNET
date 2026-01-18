package sk.ainet.io.gguf.llama

import org.junit.Test
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.FloatBufferTensorData
import sk.ainet.lang.tensor.data.MmapFloatTensorData
import sk.ainet.lang.tensor.data.MmapTensorSource
import sk.ainet.lang.types.FP32
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MmapTensorDataTest {

    @Test
    fun `MmapFloatTensorData provides correct element access`() {
        // Create a temporary file with known float values
        val tempFile = Files.createTempFile("mmap_test_", ".bin")
        try {
            val testData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
            RandomAccessFile(tempFile.toFile(), "rw").use { raf ->
                val buffer = ByteBuffer.allocate(testData.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                testData.forEach { buffer.putFloat(it) }
                raf.write(buffer.array())
            }

            // Memory-map the file and create tensor data
            val channel = RandomAccessFile(tempFile.toFile(), "r").channel
            try {
                val mmapSource = MmapTensorSource.fromChannel(channel)
                val shape = Shape(2, 3) // 2x3 matrix
                val tensorData = mmapSource.floatTensorAt<FP32>(0, shape)

                // Verify shape
                assertEquals(shape, tensorData.shape)

                // Verify element access
                assertEquals(1.0f, tensorData[0, 0], 0.001f)
                assertEquals(2.0f, tensorData[0, 1], 0.001f)
                assertEquals(3.0f, tensorData[0, 2], 0.001f)
                assertEquals(4.0f, tensorData[1, 0], 0.001f)
                assertEquals(5.0f, tensorData[1, 1], 0.001f)
                assertEquals(6.0f, tensorData[1, 2], 0.001f)

                mmapSource.close()
            } finally {
                channel.close()
            }
        } finally {
            Files.deleteIfExists(tempFile)
        }
    }

    @Test
    fun `MmapFloatTensorData implements FloatBufferTensorData`() {
        val tempFile = Files.createTempFile("mmap_interface_test_", ".bin")
        try {
            val testData = floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f)
            RandomAccessFile(tempFile.toFile(), "rw").use { raf ->
                val buffer = ByteBuffer.allocate(testData.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                testData.forEach { buffer.putFloat(it) }
                raf.write(buffer.array())
            }

            val channel = RandomAccessFile(tempFile.toFile(), "r").channel
            try {
                val mmapSource = MmapTensorSource.fromChannel(channel)
                val tensorData = mmapSource.floatTensorAt<FP32>(0, Shape(4))

                // Verify implements FloatBufferTensorData
                assertTrue(tensorData is FloatBufferTensorData<*>)

                // Verify buffer access
                val floatBuffer = (tensorData as FloatBufferTensorData<*>).floatBuffer
                assertEquals(10.0f, floatBuffer.get(0), 0.001f)
                assertEquals(20.0f, floatBuffer.get(1), 0.001f)
                assertEquals(30.0f, floatBuffer.get(2), 0.001f)
                assertEquals(40.0f, floatBuffer.get(3), 0.001f)

                mmapSource.close()
            } finally {
                channel.close()
            }
        } finally {
            Files.deleteIfExists(tempFile)
        }
    }

    @Test
    fun `MmapTensorSource can create multiple tensor views at different offsets`() {
        val tempFile = Files.createTempFile("mmap_multi_tensor_", ".bin")
        try {
            // Write two tensors: first 4 floats, then 4 more floats
            val tensor1Data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
            val tensor2Data = floatArrayOf(100.0f, 200.0f, 300.0f, 400.0f)
            RandomAccessFile(tempFile.toFile(), "rw").use { raf ->
                val buffer = ByteBuffer.allocate((tensor1Data.size + tensor2Data.size) * 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                tensor1Data.forEach { buffer.putFloat(it) }
                tensor2Data.forEach { buffer.putFloat(it) }
                raf.write(buffer.array())
            }

            val channel = RandomAccessFile(tempFile.toFile(), "r").channel
            try {
                val mmapSource = MmapTensorSource.fromChannel(channel)

                // Create two tensor views at different offsets
                val tensor1 = mmapSource.floatTensorAt<FP32>(0, Shape(2, 2))
                val tensor2 = mmapSource.floatTensorAt<FP32>(16, Shape(2, 2)) // offset = 4 floats * 4 bytes

                // Verify tensor1
                assertEquals(1.0f, tensor1[0, 0], 0.001f)
                assertEquals(4.0f, tensor1[1, 1], 0.001f)

                // Verify tensor2
                assertEquals(100.0f, tensor2[0, 0], 0.001f)
                assertEquals(400.0f, tensor2[1, 1], 0.001f)

                mmapSource.close()
            } finally {
                channel.close()
            }
        } finally {
            Files.deleteIfExists(tempFile)
        }
    }

    @Test
    fun `MmapFloatTensorData supports 1D vector shape`() {
        val tempFile = Files.createTempFile("mmap_1d_test_", ".bin")
        try {
            val testData = floatArrayOf(7.0f, 8.0f, 9.0f)
            RandomAccessFile(tempFile.toFile(), "rw").use { raf ->
                val buffer = ByteBuffer.allocate(testData.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                testData.forEach { buffer.putFloat(it) }
                raf.write(buffer.array())
            }

            val channel = RandomAccessFile(tempFile.toFile(), "r").channel
            try {
                val mmapSource = MmapTensorSource.fromChannel(channel)
                val tensorData = mmapSource.floatTensorAt<FP32>(0, Shape(3))

                assertEquals(7.0f, tensorData[0], 0.001f)
                assertEquals(8.0f, tensorData[1], 0.001f)
                assertEquals(9.0f, tensorData[2], 0.001f)

                mmapSource.close()
            } finally {
                channel.close()
            }
        } finally {
            Files.deleteIfExists(tempFile)
        }
    }

    @Test
    fun `MmapFloatTensorData supports 3D tensor shape`() {
        val tempFile = Files.createTempFile("mmap_3d_test_", ".bin")
        try {
            // 2x2x2 tensor = 8 floats
            val testData = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
            RandomAccessFile(tempFile.toFile(), "rw").use { raf ->
                val buffer = ByteBuffer.allocate(testData.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                testData.forEach { buffer.putFloat(it) }
                raf.write(buffer.array())
            }

            val channel = RandomAccessFile(tempFile.toFile(), "r").channel
            try {
                val mmapSource = MmapTensorSource.fromChannel(channel)
                val tensorData = mmapSource.floatTensorAt<FP32>(0, Shape(2, 2, 2))

                // Row-major order: [0,0,0]=1, [0,0,1]=2, [0,1,0]=3, [0,1,1]=4, ...
                assertEquals(1f, tensorData[0, 0, 0], 0.001f)
                assertEquals(2f, tensorData[0, 0, 1], 0.001f)
                assertEquals(3f, tensorData[0, 1, 0], 0.001f)
                assertEquals(4f, tensorData[0, 1, 1], 0.001f)
                assertEquals(5f, tensorData[1, 0, 0], 0.001f)
                assertEquals(6f, tensorData[1, 0, 1], 0.001f)
                assertEquals(7f, tensorData[1, 1, 0], 0.001f)
                assertEquals(8f, tensorData[1, 1, 1], 0.001f)

                mmapSource.close()
            } finally {
                channel.close()
            }
        } finally {
            Files.deleteIfExists(tempFile)
        }
    }
}
