package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals

class DenseTensorDataFactoryProfilingTest {
    private val factory = DenseTensorDataFactory()

    @Test
    fun profileInitSmall() {
        val shape = Shape(10, 10)
        val data = factory.init(shape, FP32::class) { indices ->
            (indices[0] + indices[1]).toFloat()
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitMedium() {
        val shape = Shape(100, 100)
        val data = factory.init(shape, FP32::class) { indices ->
            (indices[0] + indices[1]).toFloat()
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitLarge() {
        // 1M elements
        val shape = Shape(1000, 1000)
        val data = factory.init(shape, FP32::class) { indices ->
            (indices[0] + indices[1]).toFloat()
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitLarge3D() {
        // 1M elements
        val shape = Shape(100, 100, 100)
        val data = factory.init(shape, FP32::class) { indices ->
            (indices[0] + indices[1] + indices[2]).toFloat()
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileZerosLarge() {
        val shape = Shape(1000, 1000)
        val data = factory.zeros<FP32, Float>(shape, FP32::class)
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileOnesLarge() {
        val shape = Shape(1000, 1000)
        val data = factory.ones<FP32, Float>(shape, FP32::class)
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileRandnLarge() {
        val shape = Shape(1000, 1000)
        val data = factory.randn<FP32, Float>(shape, FP32::class, 0.0f, 1.0f, kotlin.random.Random(42))
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitInt8() {
        val shape = Shape(500, 500)
        val data = factory.init(shape, Int8::class) { indices ->
            (indices[0] % 127).toInt()
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitInt32() {
        val shape = Shape(500, 500)
        val data = factory.init(shape, Int32::class) { indices ->
            indices[0] * indices[1]
        }
        assertEquals(shape, data.shape)
    }

    @Test
    fun profileInitFP16() {
        val shape = Shape(500, 500)
        val data = factory.init(shape, FP16::class) { indices ->
            (indices[0] * indices[1]).toFloat()
        }
        assertEquals(shape, data.shape)
    }
}
