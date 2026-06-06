package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportComponentRole
import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

class StableHloGraphExportTest {

    @Test
    fun graphExportConverterPreservesStableHloOutput() {
        val graph = simpleAddGraph()
        val directConverter = StableHloConverterFactory.createBasic()
        val expected = directConverter.convert(graph, "shared_export")
        val context = GraphExportContext(
            backendName = StableHloExportArchitecture.backendName,
            targetName = "shared_export"
        )

        val actual = StableHloGraphExportConverter(converter = directConverter).convert(graph, context)

        assertEquals(expected.content, actual.content)
        assertEquals(expected.functionName, actual.functionName)
        assertFalse(context.diagnosticReport().hasErrors)
        assertTrue(context.diagnostics.any { it.code == "stablehlo.lowering.started" })
        assertTrue(context.diagnostics.any { it.code == "stablehlo.lowering.completed" })
    }

    @Test
    fun textWriterRecordsLogicalStableHloArtifact() {
        val module = toStableHlo(simpleAddGraph(), "text_export")
        val context = GraphExportContext(backendName = StableHloExportArchitecture.backendName)
        val writer = StableHloTextWriter(logicalPath = "build/generated/text_export.mlir")

        val text = writer.write(module, context)

        assertEquals(module.content, text)
        assertEquals(1, context.artifacts.size)
        assertEquals("build/generated/text_export.mlir", context.artifacts.single().path)
        assertEquals(GraphExportArtifactRole.SOURCE, context.artifacts.single().role)
        assertTrue(context.diagnostics.any { it.stage == GraphExportStage.WRITING })
    }

    @Test
    fun graphExporterReturnsSharedResultEnvelopeForStableHloText() {
        val graph = simpleAddGraph()
        val expected = StableHloConverterFactory.createBasic().convert(graph, "result_export")
        val context = GraphExportContext(
            backendName = StableHloExportArchitecture.backendName,
            targetName = "result_export"
        )
        val exporter = StableHloGraphExporter(
            writer = StableHloTextWriter(logicalPath = "result_export.mlir")
        )

        val result = exporter.exportText(graph, context)

        assertEquals(GraphExportStatus.SUCCESS, result.status)
        assertEquals(expected.content, result.requireSuccess())
        assertFalse(result.diagnostics.hasErrors)
        assertEquals(1, result.artifacts.size)
        assertEquals("result_export.mlir", result.artifacts.single().path)
        assertEquals(
            "StableHloConverter",
            StableHloExportArchitecture.componentNames[GraphExportComponentRole.CONVERTER]
        )
        assertEquals(
            "StableHloTextWriter",
            StableHloExportArchitecture.componentNames[GraphExportComponentRole.WRITER]
        )
    }

    private fun simpleAddGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(2, 3), "FP32"))
        )
        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)
        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))
        return graph
    }
}
