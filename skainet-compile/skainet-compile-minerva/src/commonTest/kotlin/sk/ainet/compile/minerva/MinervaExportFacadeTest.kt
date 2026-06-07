package sk.ainet.compile.minerva

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.lang.graph.DefaultComputeGraph

class MinervaExportFacadeTest {

    @Test
    fun createsFacadeAndDefaultOptions() {
        val facade = MinervaExportFacade()
        val options = minervaTestOptions()

        assertEquals(MinervaExportBackend.backendName, facade.backendName)
        assertEquals(MinervaExportBackend.backendName, facade.graphCanonicalizer.backendName)
        assertEquals(MinervaExportBackend.backendName, facade.npzWriter.backendName)
        assertEquals(MinervaTarget.ATMEGA328P, options.target)
        assertEquals(MinervaQuantization.Q8, options.quantization)
        assertEquals("jvm-sequential-mlp-q8", options.toMetadata()["phaseOneScope"])
    }

    @Test
    fun rejectsInvalidOptionsWithClearMessages() {
        val outputError = assertFailsWith<IllegalArgumentException> {
            minervaTestOptions(outputDir = "")
        }
        assertTrue(outputError.message?.contains("outputDir cannot be blank") == true)

        val projectError = assertFailsWith<IllegalArgumentException> {
            minervaTestOptions(projectName = "nested/project")
        }
        assertTrue(projectError.message?.contains("simple project directory name") == true)
    }

    @Test
    fun exportGraphRejectsEmptyGraphBeforePlaceholderStage() {
        val result = MinervaExportFacade().exportGraph(DefaultComputeGraph(), minervaTestOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertFalse(result.succeeded)
        assertEquals(MinervaExportFailureKind.COMPATIBILITY_VALIDATION_FAILED, result.failure?.kind)
        assertEquals(MinervaCompatibilityIssueKind.GRAPH_VALIDATION, result.compatibilityReport?.issues?.first()?.kind)
        assertTrue(result.diagnostics.hasErrors)
        assertTrue(result.failure?.message?.contains("at least one graph node") == true)
    }

    @Test
    fun exportGraphReturnsNotImplementedForValidatedGraph() {
        val result = MinervaExportFacade().exportGraph(validMinervaMlpGraph(), minervaTestOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.NOT_IMPLEMENTED, result.failure?.kind)
        assertEquals("minerva.export.not_implemented", result.failure?.code)
        assertEquals("#694", result.failure?.details?.get("issue"))
        assertTrue(result.diagnostics.infos.any { it.code == "minerva.graph.validation.passed" })
        assertTrue(result.diagnostics.infos.any { it.code == "minerva.lowering.completed" })
        assertTrue(result.diagnostics.infos.any { it.code == "minerva.npz.completed" })
        assertTrue(result.compatibilityReport?.compatible == true)
        assertEquals(1, result.intermediate?.layerCount)
        assertTrue(assertNotNull(result.npzModel).bytes.isNotEmpty())
        assertEquals("model.npz", result.artifacts.single { it.role == GraphExportArtifactRole.INTERMEDIATE }.path)
        assertTrue(result.metadata["target"] == MinervaTarget.ATMEGA328P.compilerId)
        assertFailsWith<IllegalStateException> {
            result.requireSuccess()
        }
    }

    @Test
    fun exportModelAcceptsComputeGraphFastPath() {
        val graph = validMinervaMlpGraph()
        val result = MinervaExportFacade().exportModel(graph, minervaTestOptions())

        assertEquals(MinervaExportFailureKind.NOT_IMPLEMENTED, result.failure?.kind)
        assertTrue(result.compatibilityReport?.compatible == true)
        assertEquals(MinervaActivation.RELU, result.intermediate?.layers?.single()?.activation)
        assertEquals(listOf("layer_0_w", "layer_0_b", "layer_0_act"), result.npzModel?.arrayNames?.filter { it.startsWith("layer_0") }?.take(3))
    }

    @Test
    fun exportModelReportsUnsupportedModelWithoutForwardPass() {
        val result = MinervaExportFacade().exportModel("not-a-graph", minervaTestOptions())

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.UNSUPPORTED_MODEL_TYPE, result.failure?.kind)
        assertTrue(result.failure?.message?.contains("forwardPass") == true)
    }

    @Test
    fun exportModelProvidesForwardPassRecordingOverload() {
        val result = MinervaExportFacade().exportModel(
            model = object {},
            forwardPass = { },
            options = minervaTestOptions(projectName = "RecordedModel")
        )

        assertEquals(GraphExportStatus.FAILED, result.status)
        val failure = assertNotNull(result.failure)
        assertTrue(
            failure.kind == MinervaExportFailureKind.COMPATIBILITY_VALIDATION_FAILED ||
                failure.kind == MinervaExportFailureKind.RECORDING_FAILED
        )
    }

    @Test
    fun exportGraphIncludesCompatibilityReportForUnsupportedGraph() {
        val result = MinervaExportFacade().exportGraph(
            graph = unsupportedMinervaOperationGraph(),
            options = minervaTestOptions()
        )

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.COMPATIBILITY_VALIDATION_FAILED, result.failure?.kind)
        val report = assertNotNull(result.compatibilityReport)
        assertFalse(report.compatible)
        assertTrue(
            report.issues.any {
                it.kind == MinervaCompatibilityIssueKind.UNSUPPORTED_OPERATION &&
                    it.nodeId == "conv" &&
                    it.operationName == "conv1d"
            }
        )
        assertEquals("conv", result.failure?.details?.get("nodeId"))
    }

    @Test
    fun exportGraphCarriesLoweredIntermediateBeforeCompilerStage() {
        val result = MinervaExportFacade().exportGraph(
            graph = validMinervaMlpGraph(),
            options = minervaTestOptions(projectName = "LoweredMlp")
        )
        val intermediate = assertNotNull(result.intermediate)

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.NOT_IMPLEMENTED, result.failure?.kind)
        assertEquals("LoweredMlp", intermediate.projectName)
        assertEquals(MinervaTensorRole.INPUT, intermediate.input.role)
        assertEquals(MinervaTensorRole.OUTPUT, intermediate.output.role)
        assertEquals("matmul", intermediate.layers.single().id)
        assertEquals("1", result.failure?.details?.get("layers"))
        assertEquals("#694", result.failure?.details?.get("issue"))
        assertEquals("model.npz", result.failure?.details?.get("npzPath"))
        assertTrue(assertNotNull(result.npzModel).bytes.isNotEmpty())
    }
}
