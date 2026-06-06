package sk.ainet.compile.hlo

import sk.ainet.compile.export.GraphExportArtifact
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportComponentRole
import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportConverter
import sk.ainet.compile.export.GraphExportResult
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportStatus
import sk.ainet.compile.export.GraphExportWriter
import sk.ainet.lang.graph.ComputeGraph

/**
 * StableHLO component mapping in the shared graph-export architecture.
 */
public object StableHloExportArchitecture {
    public const val backendName: String = "stablehlo"

    public val componentNames: Map<GraphExportComponentRole, String> = mapOf(
        GraphExportComponentRole.CONVERTER to "StableHloConverter",
        GraphExportComponentRole.CONTEXT to "ConversionContext",
        GraphExportComponentRole.REGISTRY to "StableHloOperationRegistry",
        GraphExportComponentRole.FACTORY to "StableHloConverterFactory",
        GraphExportComponentRole.WRITER to "StableHloTextWriter",
        GraphExportComponentRole.VERIFIER to "MlirValidator"
    )
}

/**
 * Adapter that exposes [StableHloConverter] through the shared export contract.
 */
public class StableHloGraphExportConverter @kotlin.jvm.JvmOverloads constructor(
    private val converter: StableHloConverter = StableHloConverterFactory.createBasic(),
    public val functionName: String = "main",
    override val backendName: String = StableHloExportArchitecture.backendName
) : GraphExportConverter<ComputeGraph, StableHloModule> {

    override fun convert(input: ComputeGraph, context: GraphExportContext): StableHloModule {
        val resolvedFunctionName = context.targetName ?: functionName
        context.info(
            stage = GraphExportStage.LOWERING,
            code = "stablehlo.lowering.started",
            message = "Lowering ComputeGraph to StableHLO MLIR.",
            details = mapOf("functionName" to resolvedFunctionName)
        )

        val module = converter.convert(input, resolvedFunctionName)

        context.info(
            stage = GraphExportStage.LOWERING,
            code = "stablehlo.lowering.completed",
            message = "Lowered ComputeGraph to StableHLO MLIR.",
            details = mapOf(
                "functionName" to module.functionName,
                "inputs" to module.inputSpecs.size.toString(),
                "outputs" to module.outputSpecs.size.toString(),
                "externalParameters" to module.externalParameters.size.toString()
            )
        )
        return module
    }
}

/**
 * Shared-contract writer that renders a [StableHloModule] as MLIR text.
 */
public class StableHloTextWriter @kotlin.jvm.JvmOverloads constructor(
    public val logicalPath: String? = null,
    override val backendName: String = StableHloExportArchitecture.backendName
) : GraphExportWriter<StableHloModule, String> {

    override fun write(intermediate: StableHloModule, context: GraphExportContext): String {
        val artifactPath = logicalPath ?: "${intermediate.functionName}.stablehlo.mlir"
        context.addArtifact(
            GraphExportArtifact(
                path = artifactPath,
                role = GraphExportArtifactRole.SOURCE,
                description = "StableHLO MLIR module text",
                metadata = mapOf(
                    "functionName" to intermediate.functionName,
                    "format" to "mlir"
                )
            )
        )
        context.info(
            stage = GraphExportStage.WRITING,
            code = "stablehlo.writing.text",
            message = "Rendered StableHLO module as MLIR text.",
            details = mapOf(
                "path" to artifactPath,
                "characters" to intermediate.content.length.toString()
            )
        )
        return intermediate.content
    }
}

/**
 * Convenience facade that composes StableHLO lowering and writing into shared results.
 */
public class StableHloGraphExporter @kotlin.jvm.JvmOverloads constructor(
    public val converter: StableHloGraphExportConverter = StableHloGraphExportConverter(),
    public val writer: StableHloTextWriter = StableHloTextWriter()
) {
    public val backendName: String
        get() = converter.backendName

    public fun exportModule(graph: ComputeGraph): GraphExportResult<StableHloModule> {
        return exportModule(graph, GraphExportContext(backendName = backendName))
    }

    public fun exportModule(
        graph: ComputeGraph,
        context: GraphExportContext
    ): GraphExportResult<StableHloModule> {
        return try {
            val module = converter.convert(graph, context)
            GraphExportResult.success(
                backendName = backendName,
                output = module,
                diagnostics = context.diagnosticReport(),
                artifacts = context.artifacts,
                metadata = context.metadata
            )
        } catch (exception: Exception) {
            stableHloFailureResult(
                backendName = backendName,
                stage = GraphExportStage.LOWERING,
                exception = exception,
                context = context
            )
        }
    }

    public fun exportText(graph: ComputeGraph): GraphExportResult<String> {
        return exportText(graph, GraphExportContext(backendName = backendName))
    }

    public fun exportText(
        graph: ComputeGraph,
        context: GraphExportContext
    ): GraphExportResult<String> {
        var stage = GraphExportStage.LOWERING
        return try {
            val module = converter.convert(graph, context)
            stage = GraphExportStage.WRITING
            val text = writer.write(module, context)
            GraphExportResult.success(
                backendName = backendName,
                output = text,
                diagnostics = context.diagnosticReport(),
                artifacts = context.artifacts,
                metadata = context.metadata
            )
        } catch (exception: Exception) {
            stableHloFailureResult(
                backendName = backendName,
                stage = stage,
                exception = exception,
                context = context
            )
        }
    }
}

private fun <T> stableHloFailureResult(
    backendName: String,
    stage: GraphExportStage,
    exception: Exception,
    context: GraphExportContext
): GraphExportResult<T> {
    val reason = exception.message ?: exception.toString()
    context.error(
        stage = stage,
        code = "stablehlo.export.failed",
        message = "StableHLO export failed: $reason"
    )
    return GraphExportResult(
        backendName = backendName,
        status = GraphExportStatus.FAILED,
        output = null,
        diagnostics = context.diagnosticReport(),
        artifacts = context.artifacts,
        metadata = context.metadata
    )
}
