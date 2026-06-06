package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.tape.Execution

/**
 * Public Minerva export facade.
 *
 * This scaffold accepts direct [ComputeGraph] inputs and exposes the same
 * traced-forward-pass shape used by other SKaiNET export facades. Real
 * compatibility validation and Minerva compiler invocation start in later
 * implementation issues.
 */
public class MinervaExportFacade @kotlin.jvm.JvmOverloads constructor(
    public val backendName: String = MinervaExportBackend.backendName,
    public val compatibilityValidator: MinervaCompatibilityValidator = MinervaCompatibilityValidator()
) {

    /**
     * Export a model when the caller may already hold a [ComputeGraph].
     */
    public fun <T : Any> exportModel(
        model: T,
        options: MinervaExportOptions
    ): MinervaExportResult {
        return when (model) {
            is ComputeGraph -> exportGraph(model, options)
            else -> unsupportedModelResult(model, options)
        }
    }

    /**
     * Export a model by recording one representative forward pass.
     */
    public fun <T : Any> exportModel(
        model: T,
        forwardPass: (ExecutionContext) -> Unit,
        options: MinervaExportOptions
    ): MinervaExportResult {
        if (model is ComputeGraph) return exportGraph(model, options)

        val context = exportContext(options)
        return try {
            val graphContext = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
            val (tape, _) = graphContext.record {
                val currentTape = this.currentTape ?: error("Failed to create a recording tape.")
                val globalStack = Execution.tapeStack
                globalStack.pushTape(currentTape)
                try {
                    forwardPass(this)
                } finally {
                    globalStack.popTape()
                }
            }
            val graph = tape?.toComputeGraph()
                ?: return recordingFailedResult(options, context, "No tape was produced during recording.")
            exportGraph(graph, options)
        } catch (exception: Exception) {
            recordingFailedResult(
                options = options,
                context = context,
                reason = exception.message ?: exception.toString()
            )
        }
    }

    /**
     * Export a [ComputeGraph] directly.
     */
    public fun exportGraph(
        graph: ComputeGraph,
        options: MinervaExportOptions
    ): MinervaExportResult {
        val context = exportContext(options)
        context.info(
            stage = GraphExportStage.CAPTURE,
            code = "minerva.graph.accepted",
            message = "Accepted ComputeGraph for Minerva export.",
            details = mapOf("nodes" to graph.nodes.size.toString())
        )

        val compatibilityReport = compatibilityValidator.validate(graph, options, context)
        if (!compatibilityReport.compatible) {
            return compatibilityValidationFailedResult(options, context, compatibilityReport)
        }

        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.NOT_IMPLEMENTED,
            stage = GraphExportStage.LOWERING,
            code = "minerva.export.not_implemented",
            message = "Minerva export passed phase-one compatibility validation; lowering, compiler invocation, packaging, and verification are implemented in follow-up issues.",
            details = mapOf(
                "nextStep" to "Implement MinervaGraphCanonicalizer",
                "issue" to "#692"
            )
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(options, context, failure, compatibilityReport)
    }

    private fun unsupportedModelResult(model: Any, options: MinervaExportOptions): MinervaExportResult {
        val context = exportContext(options)
        val typeName = model::class.simpleName ?: "unknown"
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.UNSUPPORTED_MODEL_TYPE,
            stage = GraphExportStage.CAPTURE,
            code = "minerva.model.unsupported_type",
            message = "Minerva export does not have a direct adapter for model type '$typeName'. Use a ComputeGraph or provide a forwardPass lambda to record one execution.",
            details = mapOf("modelType" to typeName)
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(options, context, failure)
    }

    private fun recordingFailedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        reason: String
    ): MinervaExportResult {
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.RECORDING_FAILED,
            stage = GraphExportStage.CAPTURE,
            code = "minerva.model.recording_failed",
            message = "Failed to record a forward pass for Minerva export: $reason"
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(options, context, failure)
    }

    private fun compatibilityValidationFailedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        report: MinervaCompatibilityReport
    ): MinervaExportResult {
        val firstIssue = report.issues.firstOrNull()
        val details = mutableMapOf(
            "issueCount" to report.issues.size.toString(),
            "target" to report.target.compilerId,
            "quantization" to report.quantization.compilerId
        )
        if (firstIssue != null) {
            details += mapOf(
                "issueKind" to firstIssue.kind.name,
                "remediation" to firstIssue.remediation
            )
            firstIssue.nodeId?.let { details["nodeId"] = it }
            firstIssue.operationName?.let { details["operationName"] = it }
            details += firstIssue.details
        }
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.COMPATIBILITY_VALIDATION_FAILED,
            stage = GraphExportStage.VALIDATION,
            code = firstIssue?.code ?: "minerva.compatibility.failed",
            message = firstIssue?.message ?: "Minerva compatibility validation failed.",
            details = details
        )
        return failedResult(options, context, failure, report)
    }

    private fun failedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        failure: MinervaExportFailure,
        compatibilityReport: MinervaCompatibilityReport? = null
    ): MinervaExportResult {
        return MinervaExportResult(
            options = options,
            status = GraphExportStatus.FAILED,
            diagnostics = context.diagnosticReport(),
            artifacts = context.artifacts,
            failure = failure,
            metadata = context.metadata,
            compatibilityReport = compatibilityReport
        )
    }

    private fun exportContext(options: MinervaExportOptions): GraphExportContext {
        return GraphExportContext(
            backendName = backendName,
            targetName = options.projectName,
            metadata = options.toMetadata()
        )
    }
}
