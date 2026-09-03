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

class DefaultCpuOpsJvmElementwiseTest {
    private val dataFactory = DenseTensorDataFactory()

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun add_sub_mul_div_relu_match_scalar_reference_with_tails() {
        val rng = Random(123)
        val species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
        val lengths = buildList {
            addAll(listOf(1, 2, 3, 7, 15))
            add(species - 1)
            add(species + 1)
            add(species * 2)
            add(species * 2 + 5)
        }
        for (n in lengths) {
            val a = FloatArray(n) { (it - n / 2).toFloat() / 5f }
            val b = FloatArray(n) { ((it % 4) - 2).toFloat() / 3f }
            val tA = floatTensor(Shape(n), a)
            val tB = floatTensor(Shape(n), b)

            val scalar = DefaultCpuOps(dataFactory)
            val jvm = DefaultCpuOpsJvm(dataFactory)

            // add
            compareTensors(scalar.add(tA, tB), jvm.add(tA, tB))
            // subtract
            compareTensors(scalar.subtract(tA, tB), jvm.subtract(tA, tB))
            // multiply
            compareTensors(scalar.multiply(tA, tB), jvm.multiply(tA, tB))
            // divide (avoid division by zero in reference array by tweaking small numbers)
            for (i in b.indices) if (b[i] == 0f) b[i] = 1f
            val tBNonZero = floatTensor(Shape(n), b)
            compareTensors(scalar.divide(tA, tBNonZero), jvm.divide(tA, tBNonZero))

            // relu
            compareTensors(scalar.relu(tA), jvm.relu(tA))
        }
    }

    @Test
    fun vector_flag_on_and_off_produce_same_results() {
        val n = 257 // ensure tail
        val a = FloatArray(n) { (it - 128).toFloat() / 9f }
        val b = FloatArray(n) { ((it % 5) - 2).toFloat() }
        val tA = floatTensor(Shape(n), a)
        val tB = floatTensor(Shape(n), b)

        // Vector ON
        System.setProperty("skainet.cpu.vector.enabled", "true")
        val jvmOps = DefaultCpuOpsJvm(dataFactory)
        val onAdd = jvmOps.add(tA, tB)
        val onRelu = jvmOps.relu(tA)

        // Vector OFF -> platform factory should return DefaultCpuOps
        System.setProperty("skainet.cpu.vector.enabled", "false")
        val scalarOps = platformDefaultCpuOpsFactory()(dataFactory, sk.ainet.context.schedule.Schedule.Sequential)
        val offAdd = scalarOps.add(tA, tB)
        val offRelu = scalarOps.relu(tA)

        compareTensors(onAdd, offAdd)
        compareTensors(onRelu, offRelu)
    }

    private fun compareTensors(a: sk.ainet.lang.tensor.Tensor<*, *>, b: sk.ainet.lang.tensor.Tensor<*, *>, eps: Float = 1e-6f) {
        val va = (a.data as FloatArrayTensorData<*>).buffer
        val vb = (b.data as FloatArrayTensorData<*>).buffer
        assertEquals(va.size, vb.size)
        for (i in va.indices) assertEquals(va[i], vb[i], eps)
    }

    private fun floatTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }
}
