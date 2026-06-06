package sk.ainet.compile.minerva

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

class MinervaExportFacadeTest {

    @Test
    fun createsFacadeAndDefaultOptions() {
        val facade = MinervaExportFacade()
        val options = testOptions()

        assertEquals(MinervaExportBackend.backendName, facade.backendName)
        assertEquals(MinervaTarget.ATMEGA328P, options.target)
        assertEquals(MinervaQuantization.Q8, options.quantization)
        assertEquals("jvm-sequential-mlp-q8", options.toMetadata()["phaseOneScope"])
    }

    @Test
    fun rejectsInvalidOptionsWithClearMessages() {
        val outputError = assertFailsWith<IllegalArgumentException> {
            testOptions(outputDir = "")
        }
        assertTrue(outputError.message?.contains("outputDir cannot be blank") == true)

        val projectError = assertFailsWith<IllegalArgumentException> {
            testOptions(projectName = "nested/project")
        }
        assertTrue(projectError.message?.contains("simple project directory name") == true)
    }

    @Test
    fun exportGraphRejectsEmptyGraphBeforePlaceholderStage() {
        val result = MinervaExportFacade().exportGraph(DefaultComputeGraph(), testOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertFalse(result.succeeded)
        assertEquals(MinervaExportFailureKind.GRAPH_VALIDATION_FAILED, result.failure?.kind)
        assertTrue(result.diagnostics.hasErrors)
        assertTrue(result.failure?.details?.values?.any { it.contains("at least one graph node") } == true)
    }

    @Test
    fun exportGraphReturnsNotImplementedForValidatedGraph() {
        val result = MinervaExportFacade().exportGraph(singleInputGraph(), testOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.NOT_IMPLEMENTED, result.failure?.kind)
        assertEquals("minerva.export.not_implemented", result.failure?.code)
        assertTrue(result.diagnostics.infos.any { it.code == "minerva.graph.validation.passed" })
        assertTrue(result.metadata["target"] == MinervaTarget.ATMEGA328P.compilerId)
        assertFailsWith<IllegalStateException> {
            result.requireSuccess()
        }
    }

    @Test
    fun exportModelAcceptsComputeGraphFastPath() {
        val graph = singleInputGraph()
        val result = MinervaExportFacade().exportModel(graph, testOptions())

        assertEquals(MinervaExportFailureKind.NOT_IMPLEMENTED, result.failure?.kind)
    }

    @Test
    fun exportModelReportsUnsupportedModelWithoutForwardPass() {
        val result = MinervaExportFacade().exportModel("not-a-graph", testOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.UNSUPPORTED_MODEL_TYPE, result.failure?.kind)
        assertTrue(result.failure?.message?.contains("forwardPass") == true)
    }

    @Test
    fun exportModelProvidesForwardPassRecordingOverload() {
        val result = MinervaExportFacade().exportModel(
            model = object {},
            forwardPass = { },
            options = testOptions(projectName = "RecordedModel")
        )

        assertEquals(GraphExportStatus.FAILED, result.status)
        val failure = assertNotNull(result.failure)
        assertTrue(
            failure.kind == MinervaExportFailureKind.GRAPH_VALIDATION_FAILED ||
                failure.kind == MinervaExportFailureKind.RECORDING_FAILED
        )
    }

    private fun testOptions(
        outputDir: String = "build/minerva",
        projectName: String = "TinyMlp"
    ): MinervaExportOptions {
        return MinervaExportOptions(
            outputDir = outputDir,
            projectName = projectName,
            metadata = mapOf("test" to "true")
        )
    }

    private fun singleInputGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        graph.addNode(
            GraphNode(
                id = "input",
                operation = InputOperation<DType, Any>(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("x", listOf(1, 4), "FP32"))
            )
        )
        return graph
    }
}
