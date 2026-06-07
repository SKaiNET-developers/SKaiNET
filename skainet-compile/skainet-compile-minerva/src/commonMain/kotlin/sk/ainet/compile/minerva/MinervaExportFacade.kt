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
    public val compatibilityValidator: MinervaCompatibilityValidator = MinervaCompatibilityValidator(),
    public val graphCanonicalizer: MinervaGraphCanonicalizer = MinervaGraphCanonicalizer(),
    public val npzWriter: MinervaNpzModelWriter = MinervaNpzModelWriter()
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

        val intermediate = try {
            graphCanonicalizer.convert(graph, context)
        } catch (exception: MinervaLoweringException) {
            return loweringFailedResult(options, context, compatibilityReport, exception)
        }

        val npzModel = try {
            npzWriter.write(intermediate, context)
        } catch (exception: MinervaNpzSchemaException) {
            return npzSchemaFailedResult(
                options = options,
                context = context,
                compatibilityReport = compatibilityReport,
                intermediate = intermediate,
                exception = exception
            )
        }

        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.NOT_IMPLEMENTED,
            stage = GraphExportStage.PACKAGING,
            code = "minerva.export.not_implemented",
            message = "Minerva export lowered the graph and emitted the NPZ compiler input; compiler invocation, packaging, and verification are implemented in follow-up issues.",
            details = mapOf(
                "nextStep" to "Invoke libminerva compiler and package generated outputs.",
                "issue" to "#694",
                "layers" to intermediate.layerCount.toString(),
                "input" to intermediate.input.id,
                "output" to intermediate.output.id,
                "npzPath" to npzModel.logicalPath,
                "npzBytes" to npzModel.bytes.size.toString()
            )
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(
            options = options,
            context = context,
            failure = failure,
            compatibilityReport = compatibilityReport,
            intermediate = intermediate,
            npzModel = npzModel
        )
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

    private fun loweringFailedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        compatibilityReport: MinervaCompatibilityReport,
        exception: MinervaLoweringException
    ): MinervaExportResult {
        val details = mutableMapOf(
            "code" to exception.code,
            "issue" to "#692"
        )
        exception.nodeId?.let { details["nodeId"] = it }
        exception.operationName?.let { details["operationName"] = it }
        details += exception.details
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.LOWERING_FAILED,
            stage = GraphExportStage.LOWERING,
            code = exception.code,
            message = exception.message ?: "Minerva graph lowering failed.",
            details = details
        )
        return failedResult(
            options = options,
            context = context,
            failure = failure,
            compatibilityReport = compatibilityReport
        )
    }

    private fun npzSchemaFailedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        compatibilityReport: MinervaCompatibilityReport,
        intermediate: MinervaIntermediate,
        exception: MinervaNpzSchemaException
    ): MinervaExportResult {
        val details = mutableMapOf(
            "code" to exception.code,
            "issue" to "#693"
        )
        exception.layerId?.let { details["layerId"] = it }
        exception.arrayName?.let { details["arrayName"] = it }
        details += exception.details
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.NPZ_SCHEMA_FAILED,
            stage = GraphExportStage.WRITING,
            code = exception.code,
            message = exception.message ?: "Minerva NPZ schema validation failed.",
            details = details
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(
            options = options,
            context = context,
            failure = failure,
            compatibilityReport = compatibilityReport,
            intermediate = intermediate
        )
    }

    private fun failedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        failure: MinervaExportFailure,
        compatibilityReport: MinervaCompatibilityReport? = null,
        intermediate: MinervaIntermediate? = null,
        npzModel: MinervaNpzModel? = null
    ): MinervaExportResult {
        return MinervaExportResult(
            options = options,
            status = GraphExportStatus.FAILED,
            diagnostics = context.diagnosticReport(),
            artifacts = context.artifacts,
            failure = failure,
            metadata = context.metadata,
            compatibilityReport = compatibilityReport,
            intermediate = intermediate,
            npzModel = npzModel
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
