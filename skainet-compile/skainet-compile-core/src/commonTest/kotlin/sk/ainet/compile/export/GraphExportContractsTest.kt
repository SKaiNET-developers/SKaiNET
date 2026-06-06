package sk.ainet.compile.export

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class GraphExportContractsTest {
    @Test
    fun diagnosticReportSeparatesSeverities() {
        val report = GraphExportDiagnosticReport.empty()
            .plus(
                GraphExportDiagnostic(
                    severity = GraphExportSeverity.INFO,
                    stage = GraphExportStage.CAPTURE,
                    code = "capture.started",
                    message = "capture started"
                )
            )
            .plus(
                GraphExportDiagnostic(
                    severity = GraphExportSeverity.WARNING,
                    stage = GraphExportStage.VALIDATION,
                    code = "validation.fallback",
                    message = "fallback path used"
                )
            )
            .plus(
                GraphExportDiagnostic(
                    severity = GraphExportSeverity.ERROR,
                    stage = GraphExportStage.LOWERING,
                    code = "lowering.unsupported",
                    message = "unsupported pattern",
                    nodeId = "n1",
                    operationName = "reshape"
                )
            )

        assertEquals(1, report.infos.size)
        assertEquals(1, report.warnings.size)
        assertEquals(1, report.errors.size)
        assertTrue(report.hasErrors)
        assertEquals("n1", report.errors.single().nodeId)
    }

    @Test
    fun diagnosticReportRequireNoErrorsThrowsForErrors() {
        val report = GraphExportDiagnosticReport(
            listOf(
                GraphExportDiagnostic(
                    severity = GraphExportSeverity.ERROR,
                    stage = GraphExportStage.VALIDATION,
                    code = "validation.unsupported",
                    message = "unsupported operation"
                )
            )
        )

        assertFailsWith<IllegalStateException> {
            report.requireNoErrors()
        }
    }

    @Test
    fun contextCollectsDiagnosticsAndArtifacts() {
        val context = GraphExportContext(
            backendName = "minerva",
            targetName = "atmega328p",
            metadata = mapOf("quantization" to "q8")
        )

        context.warning(
            stage = GraphExportStage.VALIDATION,
            code = "validation.experimental",
            message = "backend is experimental"
        )
        context.addArtifact(
            GraphExportArtifact(
                path = "build/minerva/model.npz",
                role = GraphExportArtifactRole.INTERMEDIATE,
                description = "Minerva compiler input"
            )
        )

        val snapshot = context.snapshot()

        assertEquals("minerva", snapshot.backendName)
        assertEquals("atmega328p", snapshot.targetName)
        assertEquals(1, snapshot.diagnostics.size)
        assertEquals(1, snapshot.artifacts.size)
        assertEquals("q8", snapshot.metadata["quantization"])
        assertFalse(context.diagnosticReport().hasErrors)
    }

    @Test
    fun resultRequireSuccessReturnsBackendOutput() {
        val result = GraphExportResult.success(
            backendName = "stablehlo",
            output = "module { }",
            artifacts = listOf(
                GraphExportArtifact(
                    path = "build/model.mlir",
                    role = GraphExportArtifactRole.SOURCE
                )
            )
        )

        assertTrue(result.succeeded)
        assertEquals("module { }", result.requireSuccess())
        assertEquals(GraphExportStatus.SUCCESS, result.status)
        assertEquals(1, result.artifacts.size)
    }

    @Test
    fun resultRequireSuccessThrowsForFailure() {
        val result = GraphExportResult.failure(
            backendName = "minerva",
            diagnostics = GraphExportDiagnosticReport(
                listOf(
                    GraphExportDiagnostic(
                        severity = GraphExportSeverity.ERROR,
                        stage = GraphExportStage.VALIDATION,
                        code = "validation.unsupported",
                        message = "unsupported graph"
                    )
                )
            )
        )

        assertTrue(result.failed)
        assertFailsWith<IllegalStateException> {
            result.requireSuccess()
        }
    }

    @Test
    fun componentRolesDocumentNamingConventions() {
        assertEquals("Converter", GraphExportComponentRole.CONVERTER.suffix)
        assertEquals("Context", GraphExportComponentRole.CONTEXT.suffix)
        assertEquals("Registry", GraphExportComponentRole.REGISTRY.suffix)
        assertEquals("Factory", GraphExportComponentRole.FACTORY.suffix)
        assertEquals("Writer", GraphExportComponentRole.WRITER.suffix)
        assertEquals("Verifier", GraphExportComponentRole.VERIFIER.suffix)
    }
}
