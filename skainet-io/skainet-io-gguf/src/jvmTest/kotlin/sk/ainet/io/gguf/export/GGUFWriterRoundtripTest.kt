package sk.ainet.io.gguf.export

import kotlinx.io.Buffer
import kotlinx.io.asSource
import kotlinx.io.buffered
import kotlinx.io.readByteArray
import org.junit.Test
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGUFWriterRoundtripTest {

    private fun stubOp(name: String): Operation = object : Operation {
        override val name: String = name
        override val type: String = "stub"
        override val parameters: Map<String, Any> = emptyMap()
        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name)
    }

    @Test
    fun roundtrip_singleTensorAndGraphMetadata() {
        val request = buildSimpleRequest()
        val (report, bytes) = GGUFWriter.writeToByteArray(request)

        assertEquals(1, report.tensorCount)
        assertTrue(bytes.isNotEmpty())

        // Read back with GGUFReader to validate header and tensor directory.
        val reader = GGUFReader(bytes.inputStream().asSource().buffered(), loadTensorData = true)
        assertNotNull(reader.fields["GGUF.version"])
        val tensorCountField = reader.fields["skainet.tensor.count"]
        assertNotNull(tensorCountField)
        val countList = tensorCountField.parts[tensorCountField.data[0]] as List<*>
        val tensorCount = (countList.first() as Number).toInt()
        assertEquals(1, tensorCount)
        assertEquals(1, reader.tensors.size)
        val tensor = reader.tensors.first()
        assertEquals("layer0.weight", tensor.name)
        assertEquals(listOf(1u, 2u), tensor.shape)
        assertNotNull(reader.fields["skainet.dtype.default"])
    }

    @Test
    fun writeToSink_matchesByteArray() {
        val request = buildSimpleRequest()
        val (report, bytes) = GGUFWriter.writeToByteArray(request)
        val buffer = Buffer()
        val streamedReport = GGUFWriter.writeToSink(request, buffer)
        val streamedBytes = buffer.readByteArray()

        assertEquals(report, streamedReport)
        assertContentEquals(bytes.toList(), streamedBytes.toList())
    }

    @Test
    fun writes_additionalDtypes() {
        val request = buildSimpleDtypeRequest()
        val buffer = Buffer()
        GGUFWriter.writeToSink(request, buffer)
        val reader = GGUFReader(buffer.readByteArray().inputStream().asSource().buffered(), loadTensorData = true)

        assertEquals(2, reader.tensors.size)
        assertEquals(GGMLQuantizationType.BF16, reader.tensors[0].tensorType)
        assertEquals(GGMLQuantizationType.I16, reader.tensors[1].tensorType)
        assertEquals(2, reader.tensors[0].data.size)
        assertEquals(2, reader.tensors[1].data.size)
        val bfValues = reader.tensors[0].data.map { (it as Number).toFloat() }
        val i16Values = reader.tensors[1].data.map { (it as Number).toInt() }
        assertEquals(1.0f, bfValues[0])
        assertEquals(-2.0f, bfValues[1])
        assertEquals(7, i16Values[0])
    }

    @Test
    fun writes_additionalDtypes_rawToggle() {
        val request = buildSimpleDtypeRequest()
        val (_, bytes) = GGUFWriter.writeToByteArray(request)
        val reader = GGUFReader(
            bytes.inputStream().asSource().buffered(),
            loadTensorData = true,
            decodeBF16ToFloat = false
        )
        val bfData = reader.tensors.first { it.name == "bf" }.data
        assertTrue(bfData.all { it is UShort })
        val words = bfData.map { (it as UShort).toInt() }
        assertEquals(listOf(0x3f80, 0xc000), words) // 1.0f and -2.0f BF16 words
    }

    private fun buildSimpleDtypeRequest(): GgufWriteRequest {
        val dataFactory = DenseTensorDataFactory()
        val bfData = dataFactory.fromFloatArray(Shape(2), floatArrayOf(1.0f, -2.0f), FP32)
        val bfTensor = VoidOpsTensor(bfData, FP32::class)
        val intData = dataFactory.full(Shape(2), 7, Int32)
        val i16Tensor = VoidOpsTensor(intData as TensorData<Int32, *>, Int32::class)
        return GgufWriteRequest(
            metadata = mapOf(
                "model.name" to "types",
                "skainet.graph.nodes" to "[]",
                "skainet.graph.edges" to "[]",
                "skainet.graph.format_version" to 1,
                "skainet.tensor.map" to "{\"bf\":\"bf\",\"i16\":\"i16\"}",
                "skainet.tensor.count" to 2
            ),
            tensors = listOf(
                GgufTensorEntry("bf", bfTensor, GGMLQuantizationType.BF16, listOf(2)),
                GgufTensorEntry("i16", i16Tensor, GGMLQuantizationType.I16, listOf(2))
            ),
            tensorMap = mapOf("bf" to "bf", "i16" to "i16")
        )
    }

    private fun buildSimpleRequest(): GgufWriteRequest {
        // Build a tiny graph: a -> b with one edge.
        val graph = DefaultComputeGraph()
        val a = graph.addNode(
            GraphNode(
                id = "n0",
                operation = stubOp("input"),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("t0", listOf(1, 2), "FP32"))
            )
        )
        val b = graph.addNode(
            GraphNode(
                id = "n1",
                operation = stubOp("relu"),
                inputs = listOf(TensorSpec("t0", listOf(1, 2), "FP32")),
                outputs = listOf(TensorSpec("t1", listOf(1, 2), "FP32"))
            )
        )
        graph.addEdge(
            GraphEdge(
                id = "e0",
                source = a,
                destination = b,
                sourceOutputIndex = 0,
                destinationInputIndex = 0,
                tensorSpec = a.outputs.first()
            )
        )

        // One FP32 weight tensor.
        val dataFactory = DenseTensorDataFactory()
        val weightData = dataFactory.fromFloatArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f), FP32)
        val weightTensor = VoidOpsTensor(weightData, FP32::class)
        val weights = mapOf("layer0.weight" to weightTensor)

        return exportGraphToGguf(graph, weights, label = "demo")
    }
}
