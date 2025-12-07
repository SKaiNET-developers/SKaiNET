package sk.ainet.io.gguf.export

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class GGUFIntegrationDslRoundtripTest {

    private val tensorFactory: TensorDataFactory = DenseTensorDataFactory()

    @Test
    fun modelWeights_roundtripThroughGguf() {
        val weights = tensorFactory.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val module = SingleWeightModule(
            name = "dummy",
            weight = VoidOpsTensor(weights, FP32::class)
        )

        val (report, bytes) = writeModelToGgufBytes(
            model = module,
            forwardPass = { /* no-op graph */ },
            label = "dummy"
        )

        assertEquals(1, report.tensorCount)

        val reader = GGUFReader(bytes.inputStream().asSource().buffered(), loadTensorData = true)
        val weightTensor = reader.tensors.first { it.name == "dummy.weight" }
        assertEquals(listOf(2u, 2u), weightTensor.shape)
        val data = weightTensor.data.map { (it as Number).toFloat() }
        assertContentEquals(listOf(1f, 2f, 3f, 4f), data)
    }

    @Test
    fun int8Weights_roundtripThroughGguf() {
        val weights = tensorFactory.fromByteArray<Int8, Byte>(Shape(3), Int8::class, byteArrayOf(1, -2, 3))
        val module = Int8Module("int8mod", VoidOpsTensor(weights, Int8::class))

        val (_, bytes) = writeModelToGgufBytes(
            model = module,
            forwardPass = { /* metadata-only graph */ },
            label = "int8mod"
        )

        val reader = GGUFReader(bytes.inputStream().asSource().buffered(), loadTensorData = true)
        val tensor = reader.tensors.first { it.name == "int8mod.weight" }
        assertEquals(listOf(3u), tensor.shape)
        val ints = tensor.data.map { (it as Number).toInt() }
        assertContentEquals(listOf(1, -2, 3), ints)
    }

    /** Minimal module with a single weight parameter; forward is unused for this export test. */
    private class SingleWeightModule(
        override val name: String,
        weight: Tensor<FP32, Float>
    ) : Module<FP32, Float>(), ModuleParameters<FP32, Float> {
        override val modules: List<Module<FP32, Float>> = emptyList()
        private val weightParam = ModuleParameter.WeightParameter("weight", weight)
        override val params: List<ModuleParameter<FP32, Float>> = listOf(weightParam)
        override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> = input
    }

    private class Int8Module(
        override val name: String,
        weight: Tensor<Int8, Byte>
    ) : Module<Int8, Byte>(), ModuleParameters<Int8, Byte> {
        override val modules: List<Module<Int8, Byte>> = emptyList()
        private val weightParam = ModuleParameter.WeightParameter("weight", weight)
        override val params: List<ModuleParameter<Int8, Byte>> = listOf(weightParam)
        override fun forward(input: Tensor<Int8, Byte>, ctx: ExecutionContext): Tensor<Int8, Byte> = input
    }
}
