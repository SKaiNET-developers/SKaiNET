package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Validates the phase-one Minerva graph contract before lowering or compiler invocation.
 */
public class MinervaCompatibilityValidator @kotlin.jvm.JvmOverloads constructor(
    public val sramSafetyBytes: Int = 0
) {
    public fun validate(
        graph: ComputeGraph,
        options: MinervaExportOptions
    ): MinervaCompatibilityReport {
        val context = GraphExportContext(
            backendName = MinervaExportBackend.backendName,
            targetName = options.projectName,
            metadata = options.toMetadata()
        )
        return validate(graph, options, context)
    }

    public fun validate(
        graph: ComputeGraph,
        options: MinervaExportOptions,
        context: GraphExportContext
    ): MinervaCompatibilityReport {
        val issues = mutableListOf<MinervaCompatibilityIssue>()
        context.info(
            stage = GraphExportStage.VALIDATION,
            code = "minerva.compatibility.started",
            message = "Started Minerva phase-one compatibility validation.",
            details = mapOf(
                "target" to options.target.compilerId,
                "quantization" to options.quantization.compilerId,
                "nodes" to graph.nodes.size.toString()
            )
        )

        if (options.quantization != MinervaQuantization.Q8) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.UNSUPPORTED_QUANTIZATION,
                code = "minerva.compatibility.unsupported_quantization",
                message = "Minerva phase one supports Q8 export only.",
                remediation = "Use MinervaQuantization.Q8 for phase-one exports.",
                details = mapOf("quantization" to options.quantization.compilerId)
            )
        }

        if (graph.nodes.isEmpty()) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.GRAPH_VALIDATION,
                code = "minerva.compatibility.empty_graph",
                message = "Minerva export requires at least one graph node.",
                remediation = "Provide a traced or constructed ComputeGraph with input and layer nodes."
            )
            return report(options, context, issues, layerCount = 0, estimatedSramBytes = 0, estimatedFlashBytes = 0)
        }

        when (val validation = graph.validate()) {
            is ValidationResult.Valid -> context.info(
                stage = GraphExportStage.VALIDATION,
                code = "minerva.graph.validation.passed",
                message = "ComputeGraph validation passed before Minerva-specific checks."
            )
            is ValidationResult.Invalid -> {
                validation.errors.forEachIndexed { index, error ->
                    recordIssue(
                        issues = issues,
                        context = context,
                        kind = MinervaCompatibilityIssueKind.GRAPH_VALIDATION,
                        code = "minerva.compatibility.graph_invalid",
                        message = error,
                        remediation = "Fix the ComputeGraph structural validation error before exporting to Minerva.",
                        details = mapOf("errorIndex" to index.toString())
                    )
                }
                return report(options, context, issues, layerCount = 0, estimatedSramBytes = 0, estimatedFlashBytes = 0)
            }
        }

        val topology = try {
            graph.getTopologicalOrder()
        } catch (exception: Exception) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY,
                code = "minerva.compatibility.topology_invalid",
                message = exception.message ?: "Unable to determine graph topological order.",
                remediation = "Use an acyclic sequential MLP graph."
            )
            emptyList()
        }

        validateOperations(topology, context, issues)
        validateStaticShapes(graph, context, issues)
        validateSequentialTopology(graph, context, issues)
        validateActivationPlacement(graph, context, issues)

        val layerCount = topology.count { isLayerOperation(it.operationName) }
        if (layerCount == 0) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY,
                code = "minerva.compatibility.no_layers",
                message = "Minerva phase-one export requires at least one dense, linear, or matmul layer.",
                remediation = "Export a sequential MLP graph with at least one supported layer."
            )
        }

        val estimatedSramBytes = estimateSramBytes(graph)
        val estimatedFlashBytes = estimateFlashBytes(graph)
        val maxSramBytes = (options.target.sramBytes - sramSafetyBytes).coerceAtLeast(0)
        if (estimatedSramBytes > maxSramBytes) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.MEMORY_BUDGET_EXCEEDED,
                code = "minerva.compatibility.sram_budget_exceeded",
                message = "Estimated activation memory exceeds ${options.target.displayName} SRAM budget.",
                remediation = "Reduce layer width, batch size, or target a Minerva configuration with more SRAM.",
                details = mapOf(
                    "estimatedSramBytes" to estimatedSramBytes.toString(),
                    "targetSramBytes" to options.target.sramBytes.toString(),
                    "sramSafetyBytes" to sramSafetyBytes.toString()
                )
            )
        }

        if (issues.isEmpty()) {
            context.info(
                stage = GraphExportStage.VALIDATION,
                code = "minerva.compatibility.passed",
                message = "ComputeGraph is compatible with the phase-one Minerva export contract.",
                details = mapOf(
                    "layerCount" to layerCount.toString(),
                    "estimatedSramBytes" to estimatedSramBytes.toString(),
                    "estimatedFlashBytes" to estimatedFlashBytes.toString()
                )
            )
        }

        return report(options, context, issues, layerCount, estimatedSramBytes, estimatedFlashBytes)
    }

    private fun validateOperations(
        nodes: List<GraphNode>,
        context: GraphExportContext,
        issues: MutableList<MinervaCompatibilityIssue>
    ) {
        nodes.forEach { node ->
            if (!isSupportedOperation(node.operationName)) {
                recordIssue(
                    issues = issues,
                    context = context,
                    kind = MinervaCompatibilityIssueKind.UNSUPPORTED_OPERATION,
                    code = "minerva.compatibility.unsupported_operation",
                    message = "Operation '${node.operationName}' is not supported by Minerva phase one.",
                    node = node,
                    remediation = "Use sequential MLP operations: matmul/dense/linear, add for bias, and relu/sigmoid/tanh activations.",
                    details = mapOf("operationType" to node.operationType)
                )
            }
        }
    }

    private fun validateStaticShapes(
        graph: ComputeGraph,
        context: GraphExportContext,
        issues: MutableList<MinervaCompatibilityIssue>
    ) {
        graph.nodes.forEach { node ->
            val specs = node.inputs.map { "input:${it.name}" to it } + node.outputs.map { "output:${it.name}" to it }
            specs.forEach { (slot, spec) ->
                val shape = spec.shape
                when {
                    shape == null -> recordIssue(
                        issues = issues,
                        context = context,
                        kind = MinervaCompatibilityIssueKind.MISSING_SHAPE,
                        code = "minerva.compatibility.missing_shape",
                        message = "Tensor '$slot' on node '${node.id}' has a dynamic or missing shape.",
                        node = node,
                        remediation = "Provide fully static tensor shapes before exporting to Minerva.",
                        details = mapOf("tensor" to slot, "dtype" to spec.dtype)
                    )
                    shape.isEmpty() || shape.any { it <= 0 } -> recordIssue(
                        issues = issues,
                        context = context,
                        kind = MinervaCompatibilityIssueKind.INVALID_SHAPE,
                        code = "minerva.compatibility.invalid_shape",
                        message = "Tensor '$slot' on node '${node.id}' has an invalid shape $shape.",
                        node = node,
                        remediation = "Use non-empty static shapes with positive dimensions.",
                        details = mapOf("tensor" to slot, "shape" to shape.joinToString("x"))
                    )
                }
            }
        }
        graph.edges.forEach { edge ->
            val shape = edge.tensorSpec.shape
            if (shape == null) {
                recordIssue(
                    issues = issues,
                    context = context,
                    kind = MinervaCompatibilityIssueKind.MISSING_SHAPE,
                    code = "minerva.compatibility.missing_edge_shape",
                    message = "Edge '${edge.id}' from '${edge.source.id}' to '${edge.destination.id}' has a dynamic or missing shape.",
                    node = edge.destination,
                    remediation = "Resolve all edge tensor shapes before exporting to Minerva.",
                    details = mapOf("edgeId" to edge.id)
                )
            } else if (shape.isEmpty() || shape.any { it <= 0 }) {
                recordIssue(
                    issues = issues,
                    context = context,
                    kind = MinervaCompatibilityIssueKind.INVALID_SHAPE,
                    code = "minerva.compatibility.invalid_edge_shape",
                    message = "Edge '${edge.id}' has an invalid shape $shape.",
                    node = edge.destination,
                    remediation = "Use non-empty static edge shapes with positive dimensions.",
                    details = mapOf("edgeId" to edge.id, "shape" to shape.joinToString("x"))
                )
            }
        }
    }

    private fun validateSequentialTopology(
        graph: ComputeGraph,
        context: GraphExportContext,
        issues: MutableList<MinervaCompatibilityIssue>
    ) {
        val outputs = graph.getOutputNodes()
        if (outputs.size != 1) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY,
                code = "minerva.compatibility.output_count",
                message = "Minerva phase one expects exactly one graph output, got ${outputs.size}.",
                remediation = "Export one sequential MLP output tensor per Minerva model.",
                details = mapOf("outputNodeIds" to outputs.joinToString(",") { it.id })
            )
        }

        graph.nodes.forEach { node ->
            val incoming = graph.edges.count { it.destination == node }
            val outgoing = graph.edges.count { it.source == node }
            if (outgoing > 1) {
                recordIssue(
                    issues = issues,
                    context = context,
                    kind = MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY,
                    code = "minerva.compatibility.branching",
                    message = "Node '${node.id}' fans out to $outgoing consumers; Minerva phase one supports sequential MLP topology only.",
                    node = node,
                    remediation = "Remove branching or split the model into separate Minerva exports.",
                    details = mapOf("consumerCount" to outgoing.toString())
                )
            }
            when (node.operationName.lowercase()) {
                "matmul" -> requireIncoming(node, incoming, 2, context, issues)
                "add" -> requireIncoming(node, incoming, 2, context, issues)
                "relu", "sigmoid", "tanh" -> requireIncoming(node, incoming, 1, context, issues)
            }
        }
    }

    private fun validateActivationPlacement(
        graph: ComputeGraph,
        context: GraphExportContext,
        issues: MutableList<MinervaCompatibilityIssue>
    ) {
        graph.nodes.filter { isActivationOperation(it.operationName) }.forEach { node ->
            val producer = graph.getInputNodes(node).singleOrNull()
            val producerName = producer?.operationName?.lowercase()
            if (producerName !in activationProducerOperations) {
                recordIssue(
                    issues = issues,
                    context = context,
                    kind = MinervaCompatibilityIssueKind.INCOMPATIBLE_ACTIVATION_PLACEMENT,
                    code = "minerva.compatibility.activation_placement",
                    message = "Activation node '${node.id}' must follow a dense, linear, matmul, or bias add layer.",
                    node = node,
                    remediation = "Place activations directly after a supported Minerva layer pattern.",
                    details = mapOf("producer" to (producerName ?: "none"))
                )
            }
        }
    }

    private fun requireIncoming(
        node: GraphNode,
        actual: Int,
        expected: Int,
        context: GraphExportContext,
        issues: MutableList<MinervaCompatibilityIssue>
    ) {
        if (actual != expected) {
            recordIssue(
                issues = issues,
                context = context,
                kind = MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY,
                code = "minerva.compatibility.arity",
                message = "Node '${node.id}' (${node.operationName}) expects $expected producer edge(s), got $actual.",
                node = node,
                remediation = "Use canonical sequential MLP fragments: matmul(input, weight), add(layer, bias), optional activation.",
                details = mapOf("expected" to expected.toString(), "actual" to actual.toString())
            )
        }
    }

    private fun estimateSramBytes(graph: ComputeGraph): Int {
        val maxTensorBytes = (graph.nodes.flatMap { it.inputs + it.outputs } + graph.edges.map { it.tensorSpec })
            .mapNotNull { spec -> spec.shape?.let { tensorBytes(it) } }
            .maxOrNull() ?: 0
        return maxTensorBytes * 2
    }

    private fun estimateFlashBytes(graph: ComputeGraph): Int {
        return (graph.nodes.flatMap { it.inputs + it.outputs } + graph.edges.map { it.tensorSpec })
            .mapNotNull { spec -> spec.shape?.let { tensorBytes(it) } }
            .sum()
    }

    private fun tensorBytes(shape: List<Int>): Int {
        return shape.fold(1) { acc, dim -> acc * dim }
    }

    private fun report(
        options: MinervaExportOptions,
        context: GraphExportContext,
        issues: List<MinervaCompatibilityIssue>,
        layerCount: Int,
        estimatedSramBytes: Int,
        estimatedFlashBytes: Int
    ): MinervaCompatibilityReport {
        return MinervaCompatibilityReport(
            compatible = issues.isEmpty(),
            diagnostics = context.diagnosticReport(),
            issues = issues.toList(),
            target = options.target,
            quantization = options.quantization,
            layerCount = layerCount,
            estimatedSramBytes = estimatedSramBytes,
            estimatedFlashBytes = estimatedFlashBytes,
            metadata = context.metadata
        )
    }

    private fun recordIssue(
        issues: MutableList<MinervaCompatibilityIssue>,
        context: GraphExportContext,
        kind: MinervaCompatibilityIssueKind,
        code: String,
        message: String,
        remediation: String,
        node: GraphNode? = null,
        details: Map<String, String> = emptyMap()
    ) {
        recordIssue(
            issues = issues,
            context = context,
            kind = kind,
            code = code,
            message = message,
            nodeId = node?.id,
            operationName = node?.operationName,
            remediation = remediation,
            details = details
        )
    }

    private fun recordIssue(
        issues: MutableList<MinervaCompatibilityIssue>,
        context: GraphExportContext,
        kind: MinervaCompatibilityIssueKind,
        code: String,
        message: String,
        nodeId: String? = null,
        operationName: String? = null,
        remediation: String,
        details: Map<String, String> = emptyMap()
    ) {
        val issue = MinervaCompatibilityIssue(
            kind = kind,
            code = code,
            message = message,
            nodeId = nodeId,
            operationName = operationName,
            remediation = remediation,
            details = details
        )
        issues += issue
        context.error(
            stage = GraphExportStage.VALIDATION,
            code = code,
            message = message,
            nodeId = nodeId,
            operationName = operationName,
            details = details + ("remediation" to remediation)
        )
    }

    private fun isSupportedOperation(operationName: String): Boolean {
        val name = operationName.lowercase()
        return name == "input" || isLayerOperation(name) || name == "add" || isActivationOperation(name)
    }

    private fun isLayerOperation(operationName: String): Boolean {
        return operationName.lowercase() in layerOperations
    }

    private fun isActivationOperation(operationName: String): Boolean {
        return operationName.lowercase() in activationOperations
    }

    public companion object {
        public val layerOperations: Set<String> = setOf("dense", "linear", "matmul")
        public val activationOperations: Set<String> = setOf("relu", "sigmoid", "tanh")
        public val activationProducerOperations: Set<String> = layerOperations + setOf("add")
    }
}
