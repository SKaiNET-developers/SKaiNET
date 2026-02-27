package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.test.runTest
import sk.ainet.compile.hlo.asInputStream
import sk.ainet.compile.hlo.asReader
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import java.io.BufferedReader
import kotlin.test.Test
import kotlin.test.assertEquals

class HloGeneratorStreamTest {

    private fun generateModule() = HloGenerator.generateBlocking(
        model = Rgb2GrayScale(),
        sampleInput = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
            .fromFloatArray<FP32, Float>(
                shape = Shape(1, 3, 4, 4),
                dtype = FP32::class,
                data = FloatArray(1 * 3 * 4 * 4) { 0.5f }
            )
    )

    @Test
    fun testContentLines() {
        val module = generateModule()
        val fromSequence = module.contentLines().toList()
        val fromString = module.content.lines()
        assertEquals(fromString, fromSequence, "contentLines() should yield the same lines as content.lines()")
    }

    @Test
    fun testAsReader() {
        val module = generateModule()
        val text = module.asReader().use { reader ->
            BufferedReader(reader).readText()
        }
        assertEquals(module.content, text, "asReader() should produce the full content")
    }

    @Test
    fun testAsInputStream() {
        val module = generateModule()
        val bytes = module.asInputStream().use { it.readBytes() }
        val text = bytes.toString(Charsets.UTF_8)
        assertEquals(module.content, text, "asInputStream() should produce UTF-8 encoded content")
    }

    @Test
    fun testGenerateBlocking() {
        val module = generateModule()
        assertEquals("main", module.functionName, "Default function name should be 'main'")
        assert(module.content.contains("module {")) { "Expected 'module {' in MLIR output" }
        assert(module.content.contains("func.func")) { "Expected 'func.func' in MLIR output" }
    }
}
