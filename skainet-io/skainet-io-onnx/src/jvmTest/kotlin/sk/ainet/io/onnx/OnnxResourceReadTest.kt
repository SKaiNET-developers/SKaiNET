package sk.ainet.io.onnx

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import onnx.ModelProto
import pbandk.decodeFromByteArray
import java.io.InputStream
import kotlin.test.Ignore

class OnnxResourceReadTest {

    @Ignore("Requires run14.onnx test fixture, which is not checked into the repository")
    @Test
    fun `read run14 onnx from resources and build graph view`() {
        val inputStream: InputStream = requireNotNull(javaClass.getResourceAsStream("/run14.onnx")) {
            "run14.onnx not found on classpath"
        }
        val bytes = inputStream.use { it.readBytes() }
        val model = ModelProto.decodeFromByteArray(bytes)

        val graph = model.graph
        assertNotNull(graph, "ONNX model should contain a GraphProto")
        assertTrue(graph.node.isNotEmpty(), "graph should contain nodes")
        assertTrue(graph.initializer.isNotEmpty(), "graph should contain initializers")

        val view = model.toGraphView()
        assertEquals(
            expected = graph.node.size,
            actual = view.nodes.size,
            message = "graph view should mirror node count"
        )
        assertEquals(
            expected = graph.initializer.size,
            actual = view.initializers.size,
            message = "graph view should mirror initializer count"
        )
    }

    @Ignore("Requires run14.onnx test fixture, which is not checked into the repository")
    @Test
    fun `run14 onnx ops are covered by importer mapping`() {
        val bytes = loadResourceBytes("run14.onnx")
        val model = ModelProto.decodeFromByteArray(bytes)
        val opTypes = model.graph?.node?.map { it.opType.uppercase() }?.toSet().orEmpty()
        val supported = setOf(
            "CONV",
            "RELU",
            "SIGMOID",
            "ADD",
            "MUL",
            "DIV",
            "SUB",
            "MATMUL",
            "GEMM",
            "MAXPOOL",
            "UPSAMPLE",
            "RESIZE",
            "RESHAPE",
            "TRANSPOSE",
            "BATCHNORMALIZATION",
            "CONCAT",
            "SOFTMAX",
            "SLICE",
            "SPLIT"
        )
        val missing = opTypes - supported
        assertTrue(missing.isEmpty(), "Unsupported ops encountered: $missing")
    }

    private fun loadResourceBytes(name: String): ByteArray {
        val inputStream: InputStream = requireNotNull(javaClass.getResourceAsStream("/$name")) {
            "$name not found on classpath"
        }
        return inputStream.use { it.readBytes() }
    }
}
