package sk.ainet.exec.tensor.ops

import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32

class DefaultCpuOpsJvmReductionsTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(dataFactory)

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun sum_all_dims_with_tail_lengths() {
        // Length that is not a multiple of preferred species to stress tail path
        val species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
        val lengths = listOf(0, 1, species - 1, species + 1, species * 2 + 3)
        for (n in lengths) {
            val arr = FloatArray(maxOf(n, 0)) { i -> (i % 7 - 3).toFloat() }
            val tensor = floatTensor(arr)
            val result = ops.sum(tensor, null)
            val expected = arr.sum()
            assertEquals(0, result.shape.rank)
            assertEquals(expected, (result.data as FloatArrayTensorData<*>).buffer[0], 1e-5f)
        }
    }

    @Test
    fun mean_all_dims_matches_reference() {
        val values = floatArrayOf(-2f, -1f, 0f, 1f, 2f, 5.5f)
        val t = floatTensor(values)
        val r = ops.mean(t, null)
        val expected = values.average().toFloat()
        assertEquals(0, r.shape.rank)
        assertEquals(expected, (r.data as FloatArrayTensorData<*>).buffer[0], 1e-6f)
    }

    @Test
    fun sum_and_mean_with_vector_flag_on_and_off() {
        val data = FloatArray(257) { i -> (i - 128).toFloat() / 3f }
        val t = floatTensor(data)

        // Run with vector ON
        System.setProperty("skainet.cpu.vector.enabled", "true")
        val rOnSum = ops.sum(t, null)
        val rOnMean = ops.mean(t, null)

        // Run with vector OFF by going through platform factory which will create DefaultCpuOps
        System.setProperty("skainet.cpu.vector.enabled", "false")
        val factory = platformDefaultCpuOpsFactory()
        val scalarOps = factory(dataFactory, sk.ainet.context.schedule.Schedule.Sequential)
        val rOffSum = scalarOps.sum(t, null)
        val rOffMean = scalarOps.mean(t, null)

        val expectedSum = data.sum()
        val expectedMean = data.average().toFloat()

        fun extractScalar(x: sk.ainet.lang.tensor.Tensor<*, *>): Float =
            (x.data as FloatArrayTensorData<*>).buffer[0]

        assertEquals(expectedSum, extractScalar(rOnSum), 1e-5f)
        assertEquals(expectedSum, extractScalar(rOffSum), 1e-5f)
        assertEquals(expectedMean, extractScalar(rOnMean), 1e-5f)
        assertEquals(expectedMean, extractScalar(rOffMean), 1e-5f)
    }

    private fun floatTensor(values: FloatArray): VoidOpsTensor<FP32, Float> {
        val shape = Shape(values.size)
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }
}
