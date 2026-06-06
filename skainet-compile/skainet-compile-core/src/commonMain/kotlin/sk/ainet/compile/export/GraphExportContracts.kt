package sk.ainet.compile.export

/**
 * Stage of a graph export workflow.
 *
 * Backends do not need to implement every stage. The enum exists so StableHLO,
 * Minerva, and later exporters can report comparable diagnostics while keeping
 * their writer implementations separate.
 */
public enum class GraphExportStage {
    CAPTURE,
    VALIDATION,
    LOWERING,
    WRITING,
    PACKAGING,
    VERIFICATION;

    public companion object {
        public val defaultOrder: List<GraphExportStage> = listOf(
            CAPTURE,
            VALIDATION,
            LOWERING,
            WRITING,
            PACKAGING,
            VERIFICATION
        )
    }
}

/**
 * Severity for export diagnostics.
 */
public enum class GraphExportSeverity {
    INFO,
    WARNING,
    ERROR
}

/**
 * Shared diagnostic record for graph exporters.
 *
 * The optional [nodeId] and [operationName] fields let backends tie a finding
 * to graph-level source information without making compile-core depend on the
 * DAG module.
 */
public data class GraphExportDiagnostic(
    public val severity: GraphExportSeverity,
    public val stage: GraphExportStage,
    public val code: String,
    public val message: String,
    public val nodeId: String? = null,
    public val operationName: String? = null,
    public val details: Map<String, String> = emptyMap()
) {
    init {
        require(code.isNotBlank()) { "Diagnostic code cannot be blank" }
        require(message.isNotBlank()) { "Diagnostic message cannot be blank" }
    }
}

/**
 * Immutable collection of export diagnostics with convenience views.
 */
public data class GraphExportDiagnosticReport(
    public val diagnostics: List<GraphExportDiagnostic> = emptyList()
) {
    public val errors: List<GraphExportDiagnostic>
        get() = diagnostics.filter { it.severity == GraphExportSeverity.ERROR }

    public val warnings: List<GraphExportDiagnostic>
        get() = diagnostics.filter { it.severity == GraphExportSeverity.WARNING }

    public val infos: List<GraphExportDiagnostic>
        get() = diagnostics.filter { it.severity == GraphExportSeverity.INFO }

    public val hasErrors: Boolean
        get() = errors.isNotEmpty()

    public operator fun plus(diagnostic: GraphExportDiagnostic): GraphExportDiagnosticReport {
        return copy(diagnostics = diagnostics + diagnostic)
    }

    public operator fun plus(other: GraphExportDiagnosticReport): GraphExportDiagnosticReport {
        return copy(diagnostics = diagnostics + other.diagnostics)
    }

    public fun requireNoErrors(): GraphExportDiagnosticReport {
        if (hasErrors) {
            val summary = errors.joinToString("; ") { "${it.code}: ${it.message}" }
            error("Graph export reported errors: $summary")
        }
        return this
    }

    public companion object {
        public fun empty(): GraphExportDiagnosticReport = GraphExportDiagnosticReport()
    }
}

/**
 * Role for a generated or consumed export artifact.
 */
public enum class GraphExportArtifactRole {
    INTERMEDIATE,
    SOURCE,
    HEADER,
    MANIFEST,
    PROJECT_DIRECTORY,
    LOG,
    TEST_REPORT,
    BINARY,
    DOCUMENTATION
}

/**
 * Portable artifact descriptor.
 *
 * Paths are strings so this type remains usable from common source sets.
 */
public data class GraphExportArtifact(
    public val path: String,
    public val role: GraphExportArtifactRole,
    public val description: String = "",
    public val sensitive: Boolean = false,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(path.isNotBlank()) { "Artifact path cannot be blank" }
    }
}

/**
 * Overall status for a backend export.
 */
public enum class GraphExportStatus {
    SUCCESS,
    FAILED,
    SKIPPED
}

/**
 * Shared result envelope for export backends.
 *
 * The [output] type is backend-specific: StableHLO can use a module object,
 * Minerva can use an export bundle, and future backends can choose their own
 * writer result while sharing diagnostics and artifact metadata.
 */
public data class GraphExportResult<T>(
    public val backendName: String,
    public val status: GraphExportStatus,
    public val output: T? = null,
    public val diagnostics: GraphExportDiagnosticReport = GraphExportDiagnosticReport.empty(),
    public val artifacts: List<GraphExportArtifact> = emptyList(),
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(backendName.isNotBlank()) { "backendName cannot be blank" }
    }

    public val succeeded: Boolean
        get() = status == GraphExportStatus.SUCCESS

    public val failed: Boolean
        get() = status == GraphExportStatus.FAILED

    public fun requireSuccess(): T {
        if (!succeeded) {
            val summary = diagnostics.diagnostics.joinToString("; ") { "${it.code}: ${it.message}" }
            error("Graph export for $backendName did not succeed: $summary")
        }
        return output ?: error("Graph export for $backendName succeeded without an output")
    }

    public companion object {
        public fun <T> success(
            backendName: String,
            output: T,
            diagnostics: GraphExportDiagnosticReport = GraphExportDiagnosticReport.empty(),
            artifacts: List<GraphExportArtifact> = emptyList(),
            metadata: Map<String, String> = emptyMap()
        ): GraphExportResult<T> {
            return GraphExportResult(
                backendName = backendName,
                status = GraphExportStatus.SUCCESS,
                output = output,
                diagnostics = diagnostics,
                artifacts = artifacts,
                metadata = metadata
            )
        }

        public fun failure(
            backendName: String,
            diagnostics: GraphExportDiagnosticReport,
            artifacts: List<GraphExportArtifact> = emptyList(),
            metadata: Map<String, String> = emptyMap()
        ): GraphExportResult<Nothing> {
            return GraphExportResult(
                backendName = backendName,
                status = GraphExportStatus.FAILED,
                output = null,
                diagnostics = diagnostics,
                artifacts = artifacts,
                metadata = metadata
            )
        }

        public fun skipped(
            backendName: String,
            reason: String,
            metadata: Map<String, String> = emptyMap()
        ): GraphExportResult<Nothing> {
            val report = GraphExportDiagnosticReport(
                listOf(
                    GraphExportDiagnostic(
                        severity = GraphExportSeverity.INFO,
                        stage = GraphExportStage.VALIDATION,
                        code = "export.skipped",
                        message = reason
                    )
                )
            )
            return GraphExportResult(
                backendName = backendName,
                status = GraphExportStatus.SKIPPED,
                output = null,
                diagnostics = report,
                metadata = metadata
            )
        }
    }
}

/**
 * Mutable context shared across validation, lowering, writing, packaging, and
 * verification phases.
 */
public class GraphExportContext(
    public val backendName: String,
    public val targetName: String? = null,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(backendName.isNotBlank()) { "backendName cannot be blank" }
    }

    private val mutableDiagnostics: MutableList<GraphExportDiagnostic> = mutableListOf()
    private val mutableArtifacts: MutableList<GraphExportArtifact> = mutableListOf()

    public val diagnostics: List<GraphExportDiagnostic>
        get() = mutableDiagnostics.toList()

    public val artifacts: List<GraphExportArtifact>
        get() = mutableArtifacts.toList()

    public fun report(diagnostic: GraphExportDiagnostic): GraphExportDiagnostic {
        mutableDiagnostics += diagnostic
        return diagnostic
    }

    public fun info(
        stage: GraphExportStage,
        code: String,
        message: String,
        nodeId: String? = null,
        operationName: String? = null,
        details: Map<String, String> = emptyMap()
    ): GraphExportDiagnostic {
        return report(
            GraphExportDiagnostic(
                severity = GraphExportSeverity.INFO,
                stage = stage,
                code = code,
                message = message,
                nodeId = nodeId,
                operationName = operationName,
                details = details
            )
        )
    }

    public fun warning(
        stage: GraphExportStage,
        code: String,
        message: String,
        nodeId: String? = null,
        operationName: String? = null,
        details: Map<String, String> = emptyMap()
    ): GraphExportDiagnostic {
        return report(
            GraphExportDiagnostic(
                severity = GraphExportSeverity.WARNING,
                stage = stage,
                code = code,
                message = message,
                nodeId = nodeId,
                operationName = operationName,
                details = details
            )
        )
    }

    public fun error(
        stage: GraphExportStage,
        code: String,
        message: String,
        nodeId: String? = null,
        operationName: String? = null,
        details: Map<String, String> = emptyMap()
    ): GraphExportDiagnostic {
        return report(
            GraphExportDiagnostic(
                severity = GraphExportSeverity.ERROR,
                stage = stage,
                code = code,
                message = message,
                nodeId = nodeId,
                operationName = operationName,
                details = details
            )
        )
    }

    public fun addArtifact(artifact: GraphExportArtifact): GraphExportArtifact {
        mutableArtifacts += artifact
        return artifact
    }

    public fun diagnosticReport(): GraphExportDiagnosticReport {
        return GraphExportDiagnosticReport(mutableDiagnostics.toList())
    }

    public fun snapshot(): GraphExportContextSnapshot {
        return GraphExportContextSnapshot(
            backendName = backendName,
            targetName = targetName,
            diagnostics = mutableDiagnostics.toList(),
            artifacts = mutableArtifacts.toList(),
            metadata = metadata
        )
    }
}

/**
 * Immutable view of a [GraphExportContext].
 */
public data class GraphExportContextSnapshot(
    public val backendName: String,
    public val targetName: String?,
    public val diagnostics: List<GraphExportDiagnostic>,
    public val artifacts: List<GraphExportArtifact>,
    public val metadata: Map<String, String> = emptyMap()
)

/**
 * Converts a source representation into a backend-specific intermediate.
 */
public interface GraphExportConverter<Input, Intermediate> {
    public val backendName: String

    public fun convert(input: Input, context: GraphExportContext): Intermediate
}

/**
 * Writes a backend-specific intermediate into its final export output.
 */
public interface GraphExportWriter<Intermediate, Output> {
    public val backendName: String

    public fun write(intermediate: Intermediate, context: GraphExportContext): Output
}

/**
 * Verification result for an exported backend output.
 */
public data class GraphExportVerification(
    public val passed: Boolean,
    public val diagnostics: GraphExportDiagnosticReport = GraphExportDiagnosticReport.empty(),
    public val artifacts: List<GraphExportArtifact> = emptyList(),
    public val metadata: Map<String, String> = emptyMap()
)

/**
 * Verifies an exported backend output.
 */
public interface GraphExportVerifier<Output> {
    public val backendName: String

    public fun verify(output: Output, context: GraphExportContext): GraphExportVerification
}

/**
 * Canonical component names for graph export implementations.
 */
public enum class GraphExportComponentRole(
    public val suffix: String,
    public val responsibility: String
) {
    CONVERTER("Converter", "Lower source graph structures into a backend intermediate."),
    CONTEXT("Context", "Carry backend state, diagnostics, and generated artifacts."),
    REGISTRY("Registry", "Map operation names or graph patterns to backend converters."),
    FACTORY("Factory", "Construct a backend exporter with standard converter registrations."),
    WRITER("Writer", "Write a backend intermediate to files, text, or model bytes."),
    VERIFIER("Verifier", "Run backend-specific validation after writing artifacts.")
}
