package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportArtifact
import sk.ainet.compile.export.GraphExportDiagnosticReport
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportStatus

/**
 * Supported Minerva quantization modes for the phase-one API.
 */
public enum class MinervaQuantization(
    public val compilerId: String
) {
    Q8("q8")
}

/**
 * Validated Minerva target configurations exposed by the export API.
 */
public enum class MinervaTarget(
    public val compilerId: String,
    public val displayName: String,
    public val flashBytes: Int,
    public val sramBytes: Int
) {
    ATMEGA328P(
        compilerId = "atmega328p",
        displayName = "ATmega328P",
        flashBytes = 32 * 1024,
        sramBytes = 2 * 1024
    )
}

/**
 * Export options for the Minerva backend.
 *
 * Path values are strings so the API stays usable from common code. The phase
 * one scaffold validates shape and intent but does not require a libminerva
 * checkout until compiler integration lands.
 */
public data class MinervaExportOptions(
    public val outputDir: String,
    public val projectName: String,
    public val target: MinervaTarget = MinervaTarget.ATMEGA328P,
    public val quantization: MinervaQuantization = MinervaQuantization.Q8,
    public val runtimeRoot: String? = null,
    public val compilerScript: String? = null,
    public val keyFile: String? = null,
    public val calibrationNpz: String? = null,
    public val dumpWeights: Boolean = false,
    public val generateHostHarness: Boolean = true,
    public val generateFirmwareExample: Boolean = true,
    public val runHostVerification: Boolean = true,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(outputDir.isNotBlank()) { "outputDir cannot be blank" }
        require(projectName.isNotBlank()) { "projectName cannot be blank" }
        require(projectName.none { it == '/' || it == '\\' }) {
            "projectName must be a simple project directory name"
        }
        requireOptionalPath("runtimeRoot", runtimeRoot)
        requireOptionalPath("compilerScript", compilerScript)
        requireOptionalPath("keyFile", keyFile)
        requireOptionalPath("calibrationNpz", calibrationNpz)
        require(metadata.keys.all { it.isNotBlank() }) { "metadata keys cannot be blank" }
    }

    public fun toMetadata(): Map<String, String> {
        return metadata + mapOf(
            "target" to target.compilerId,
            "quantization" to quantization.compilerId,
            "phaseOneScope" to MinervaExportBackend.phaseOneScope,
            "generateHostHarness" to generateHostHarness.toString(),
            "generateFirmwareExample" to generateFirmwareExample.toString(),
            "runHostVerification" to runHostVerification.toString(),
            "dumpWeights" to dumpWeights.toString()
        )
    }

    private fun requireOptionalPath(field: String, value: String?) {
        require(value == null || value.isNotBlank()) { "$field cannot be blank when provided" }
    }
}

/**
 * Stable categories for Minerva export failures.
 */
public enum class MinervaExportFailureKind {
    UNSUPPORTED_MODEL_TYPE,
    RECORDING_FAILED,
    GRAPH_VALIDATION_FAILED,
    COMPATIBILITY_VALIDATION_FAILED,
    LOWERING_FAILED,
    NOT_IMPLEMENTED
}

/**
 * Typed failure detail carried by [MinervaExportResult].
 */
public data class MinervaExportFailure(
    public val kind: MinervaExportFailureKind,
    public val stage: GraphExportStage,
    public val code: String,
    public val message: String,
    public val details: Map<String, String> = emptyMap()
) {
    init {
        require(code.isNotBlank()) { "failure code cannot be blank" }
        require(message.isNotBlank()) { "failure message cannot be blank" }
    }
}

/**
 * Future successful output bundle for a Minerva export.
 */
public data class MinervaExportBundle(
    public val projectName: String,
    public val outputDir: String,
    public val target: MinervaTarget,
    public val quantization: MinervaQuantization,
    public val generatedFiles: List<String> = emptyList(),
    public val manifestPath: String? = null
) {
    init {
        require(projectName.isNotBlank()) { "projectName cannot be blank" }
        require(outputDir.isNotBlank()) { "outputDir cannot be blank" }
        require(generatedFiles.all { it.isNotBlank() }) { "generatedFiles cannot contain blank paths" }
        require(manifestPath == null || manifestPath.isNotBlank()) {
            "manifestPath cannot be blank when provided"
        }
    }
}

/**
 * Stable categories for Minerva compatibility findings.
 */
public enum class MinervaCompatibilityIssueKind {
    GRAPH_VALIDATION,
    UNSUPPORTED_OPERATION,
    UNSUPPORTED_TOPOLOGY,
    MISSING_SHAPE,
    INVALID_SHAPE,
    INCOMPATIBLE_ACTIVATION_PLACEMENT,
    MEMORY_BUDGET_EXCEEDED,
    UNSUPPORTED_QUANTIZATION
}

/**
 * Backend-specific compatibility issue that also appears as a graph-export diagnostic.
 */
public data class MinervaCompatibilityIssue(
    public val kind: MinervaCompatibilityIssueKind,
    public val code: String,
    public val message: String,
    public val nodeId: String? = null,
    public val operationName: String? = null,
    public val remediation: String,
    public val details: Map<String, String> = emptyMap()
) {
    init {
        require(code.isNotBlank()) { "compatibility issue code cannot be blank" }
        require(message.isNotBlank()) { "compatibility issue message cannot be blank" }
        require(remediation.isNotBlank()) { "compatibility issue remediation cannot be blank" }
    }
}

/**
 * Phase-one Minerva compatibility report.
 */
public data class MinervaCompatibilityReport(
    public val compatible: Boolean,
    public val diagnostics: GraphExportDiagnosticReport,
    public val issues: List<MinervaCompatibilityIssue>,
    public val target: MinervaTarget,
    public val quantization: MinervaQuantization,
    public val layerCount: Int,
    public val estimatedSramBytes: Int,
    public val estimatedFlashBytes: Int,
    public val metadata: Map<String, String> = emptyMap()
) {
    public val failed: Boolean
        get() = !compatible

    public fun requireCompatible(): MinervaCompatibilityReport {
        if (!compatible) {
            val summary = issues.joinToString("; ") { "${it.code}: ${it.message}" }
            error("Minerva compatibility validation failed: $summary")
        }
        return this
    }
}

/**
 * Public result shape for Minerva export attempts.
 */
public data class MinervaExportResult(
    public val options: MinervaExportOptions,
    public val status: GraphExportStatus,
    public val bundle: MinervaExportBundle? = null,
    public val diagnostics: GraphExportDiagnosticReport = GraphExportDiagnosticReport.empty(),
    public val artifacts: List<GraphExportArtifact> = emptyList(),
    public val failure: MinervaExportFailure? = null,
    public val metadata: Map<String, String> = emptyMap(),
    public val compatibilityReport: MinervaCompatibilityReport? = null,
    public val intermediate: MinervaIntermediate? = null
) {
    init {
        require(status != GraphExportStatus.SUCCESS || bundle != null) {
            "Successful Minerva exports must include a bundle"
        }
        require(status == GraphExportStatus.SUCCESS || failure != null) {
            "Failed or skipped Minerva exports must include a failure"
        }
    }

    public val succeeded: Boolean
        get() = status == GraphExportStatus.SUCCESS

    public val failed: Boolean
        get() = status == GraphExportStatus.FAILED

    public fun requireSuccess(): MinervaExportBundle {
        if (!succeeded) {
            val reason = failure?.message ?: "unknown failure"
            error("Minerva export did not succeed: $reason")
        }
        return bundle ?: error("Minerva export succeeded without a bundle")
    }
}
