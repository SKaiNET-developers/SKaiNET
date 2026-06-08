package sk.ainet.compile.minerva.examples

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.compile.minerva.MinervaActivation
import sk.ainet.compile.minerva.MinervaCompatibilityValidator
import sk.ainet.compile.minerva.MinervaExportFacade
import sk.ainet.compile.minerva.MinervaExportFailureKind
import sk.ainet.compile.minerva.MinervaHostVerificationMetadata

class MinervaSecureMcuExportSamplesTest {

    @Test
    fun sensorClassifierMatchesLibminervaAtmegaDemoShape() {
        val scenario = MinervaSecureMcuExportSamples.sensorClassifier()
        val options = MinervaSecureMcuExportSamples.exportOptions(scenario)
        val report = MinervaCompatibilityValidator().validate(scenario.graph, options)

        assertTrue(report.compatible, report.issues.joinToString { it.message })
        assertEquals(3, report.layerCount)

        val result = MinervaExportFacade().exportGraph(scenario.graph, options)

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.COMPILER_PREREQUISITE_FAILED, result.failure?.kind)
        val intermediate = assertNotNull(result.intermediate)
        assertEquals(listOf(1, 8), intermediate.input.shape)
        assertEquals(listOf(1, 4), intermediate.output.shape)
        assertEquals(
            listOf(MinervaActivation.RELU, MinervaActivation.RELU, MinervaActivation.SIGMOID),
            intermediate.layers.map { it.activation }
        )
        assertEquals("idle|warmup|nominal|service", options.metadata["classLabels"])
        assertTrue(assertNotNull(result.npzModel).bytes.isNotEmpty())
        assertTrue(result.artifacts.any { it.role == GraphExportArtifactRole.INTERMEDIATE && it.path == "model.npz" })
    }

    @Test
    fun safetyGuardUsesDifferentInputAndOutputContract() {
        val scenario = MinervaSecureMcuExportSamples.safetyGuard()
        val options = MinervaSecureMcuExportSamples.exportOptions(scenario)
        val result = MinervaExportFacade().exportGraph(scenario.graph, options)

        assertEquals(GraphExportStatus.FAILED, result.status)
        assertEquals(MinervaExportFailureKind.COMPILER_PREREQUISITE_FAILED, result.failure?.kind)
        val intermediate = assertNotNull(result.intermediate)
        assertEquals(3, intermediate.layerCount)
        assertEquals(listOf(1, 6), intermediate.input.shape)
        assertEquals(listOf(1, 3), intermediate.output.shape)
        assertEquals(
            listOf(MinervaActivation.TANH, MinervaActivation.RELU, MinervaActivation.SIGMOID),
            intermediate.layers.map { it.activation }
        )
        assertEquals("protect|warn|allow", options.metadata["classLabels"])
    }

    @Test
    fun exampleOptionsCarryRealRuntimeMetadata() {
        val scenario = MinervaSecureMcuExportSamples.sensorClassifier()
        val options = MinervaSecureMcuExportSamples.exportOptions(
            scenario,
            env = mapOf(
                "MINERVA_COMPILER_SCRIPT" to "/opt/libminerva/compiler/minerva_compile.py",
                "MINERVA_RUNTIME_ROOT" to "/opt/libminerva",
                "MINERVA_KEY_FILE" to "/secure/project/device.key",
                "MINERVA_CALIBRATION_NPZ" to "/secure/project/calibration.npz",
                "MINERVA_RUN_CMAKE" to "true",
                "MINERVA_RUN_CTEST" to "true",
                "MINERVA_HOST_TOLERANCE" to "0.8",
                "MINERVA_HOST_OUTPUT_PATH" to "host-output.txt",
                "MINERVA_HOST_ADAPTER_SOURCE" to "/project/minerva_adapter.c",
                "MINERVA_HOST_INCLUDE_DIRS" to "/project/minerva-secrets",
                "MINERVA_HOST_LIBRARY_DIRS" to "/opt/libminerva/lib",
                "MINERVA_HOST_LIBRARIES" to "minerva"
            )
        )

        assertEquals("/opt/libminerva/compiler/minerva_compile.py", options.compilerScript)
        assertEquals("/opt/libminerva", options.runtimeRoot)
        assertEquals("/secure/project/device.key", options.keyFile)
        assertEquals("/secure/project/calibration.npz", options.calibrationNpz)
        assertEquals(0.8f, options.hostVerificationTolerance)
        assertEquals("minerva-sensor-classifier", options.metadata["sample"])
        assertEquals("true", options.metadata[MinervaHostVerificationMetadata.RUN_CMAKE_BUILD])
        assertEquals("true", options.metadata[MinervaHostVerificationMetadata.RUN_CTEST])
        assertEquals("host-output.txt", options.metadata[MinervaHostVerificationMetadata.HOST_OUTPUT_PATH])
        assertEquals("/project/minerva_adapter.c", options.metadata[MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE])
        assertEquals("/project/minerva-secrets", options.metadata[MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS])
        assertEquals("/opt/libminerva/lib", options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS])
        assertEquals("minerva", options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARIES])
    }
}
