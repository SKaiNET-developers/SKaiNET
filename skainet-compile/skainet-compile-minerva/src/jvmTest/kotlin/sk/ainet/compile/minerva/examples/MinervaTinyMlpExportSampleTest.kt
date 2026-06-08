package sk.ainet.compile.minerva.examples

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.compile.minerva.MinervaCompatibilityValidator
import sk.ainet.compile.minerva.MinervaExportFacade
import sk.ainet.compile.minerva.MinervaExportFailureKind
import sk.ainet.compile.minerva.MinervaHostVerificationMetadata

class MinervaTinyMlpExportSampleTest {

    @Test
    fun sampleGraphIsCompatibleAndLowersToNpz() {
        val graph = MinervaTinyMlpExportSample.tinyMlpGraph()
        val options = MinervaTinyMlpExportSample.exportOptions(
            compilerScript = "/opt/libminerva/tools/compile_model.py",
            runtimeRoot = "/opt/libminerva",
            keyFile = "/secure/project/device.key",
            calibrationNpz = "/secure/project/calibration.npz"
        )
        val report = MinervaCompatibilityValidator().validate(graph, options)

        assertTrue(report.compatible, report.issues.joinToString { it.message })
        assertEquals(2, report.layerCount)

        val dryRunOptions = MinervaTinyMlpExportSample.exportOptions()
        val result = MinervaExportFacade().exportGraph(graph, dryRunOptions)

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.COMPILER_PREREQUISITE_FAILED, result.failure?.kind)
        assertTrue(assertNotNull(result.npzModel).bytes.isNotEmpty())
        assertEquals(2, result.intermediate?.layerCount)
        assertTrue(
            result.artifacts.any {
                it.role == GraphExportArtifactRole.INTERMEDIATE && it.path == "model.npz"
            }
        )
    }

    @Test
    fun sampleOptionsCarryHostVerificationMetadata() {
        val options = MinervaTinyMlpExportSample.exportOptions(
            compilerScript = "/opt/libminerva/tools/compile_model.py",
            runCmakeBuild = true,
            runCTest = true,
            hostOutputPath = "host-output.txt"
        )

        assertEquals("minerva-tiny-mlp", options.metadata["sample"])
        assertEquals("true", options.metadata[MinervaHostVerificationMetadata.RUN_CMAKE_BUILD])
        assertEquals("true", options.metadata[MinervaHostVerificationMetadata.RUN_CTEST])
        assertEquals("host-output.txt", options.metadata[MinervaHostVerificationMetadata.HOST_OUTPUT_PATH])
    }
}
