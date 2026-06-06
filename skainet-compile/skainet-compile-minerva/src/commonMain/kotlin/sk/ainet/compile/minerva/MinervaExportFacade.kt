package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.ops.ValidationResult
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
    public val backendName: String = MinervaExportBackend.backendName
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

        if (graph.nodes.isEmpty()) {
            return graphValidationFailedResult(
                options = options,
                context = context,
                errors = listOf("Minerva export requires at least one graph node.")
            )
        }

        when (val validation = graph.validate()) {
            is ValidationResult.Valid -> context.info(
                stage = GraphExportStage.VALIDATION,
                code = "minerva.graph.validation.passed",
                message = "ComputeGraph validation passed before Minerva-specific checks."
            )
            is ValidationResult.Invalid -> return graphValidationFailedResult(
                options = options,
                context = context,
                errors = validation.errors
            )
        }

        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.NOT_IMPLEMENTED,
            stage = GraphExportStage.LOWERING,
            code = "minerva.export.not_implemented",
            message = "Minerva export API is scaffolded; compatibility validation, lowering, compiler invocation, packaging, and verification are implemented in follow-up issues.",
            details = mapOf(
                "nextStep" to "Implement MinervaCompatibilityValidator",
                "issue" to "#691"
            )
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(options, context, failure)
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

    private fun graphValidationFailedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        errors: List<String>
    ): MinervaExportResult {
        val failure = MinervaExportFailure(
            kind = MinervaExportFailureKind.GRAPH_VALIDATION_FAILED,
            stage = GraphExportStage.VALIDATION,
            code = "minerva.graph.validation_failed",
            message = "ComputeGraph validation failed before Minerva-specific checks.",
            details = errors.mapIndexed { index, error -> "error$index" to error }.toMap()
        )
        context.error(
            stage = failure.stage,
            code = failure.code,
            message = failure.message,
            details = failure.details
        )
        return failedResult(options, context, failure)
    }

    private fun failedResult(
        options: MinervaExportOptions,
        context: GraphExportContext,
        failure: MinervaExportFailure
    ): MinervaExportResult {
        return MinervaExportResult(
            options = options,
            status = GraphExportStatus.FAILED,
            diagnostics = context.diagnosticReport(),
            artifacts = context.artifacts,
            failure = failure,
            metadata = context.metadata
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
