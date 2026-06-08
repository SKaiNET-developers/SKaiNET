package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportConverter
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Registry for graph fragments that can be lowered into phase-one Minerva IR.
 */
public class MinervaLayerPatternRegistry @kotlin.jvm.JvmOverloads constructor(
    layerOperations: Set<String> = setOf("matmul", "dense", "linear"),
    public val biasOperation: String = "add",
    activationOperations: Map<String, MinervaActivation> = mapOf(
        "relu" to MinervaActivation.RELU,
        "sigmoid" to MinervaActivation.SIGMOID,
        "tanh" to MinervaActivation.TANH
    )
) {
    public val layerOperations: Set<String> = layerOperations.map { it.lowercase() }.toSet()
    public val activationOperations: Map<String, MinervaActivation> =
        activationOperations.mapKeys { it.key.lowercase() }

    init {
        require(this.layerOperations.isNotEmpty()) { "layerOperations cannot be empty" }
        require(this.layerOperations.all { it.isNotBlank() }) { "layerOperations cannot contain blanks" }
        require(biasOperation.isNotBlank()) { "biasOperation cannot be blank" }
        require(this.activationOperations.isNotEmpty()) { "activationOperations cannot be empty" }
        require(this.activationOperations.keys.all { it.isNotBlank() }) {
            "activationOperations cannot contain blank operation names"
        }
    }

    public fun isLayerOperation(operationName: String): Boolean {
        return operationName.lowercase() in layerOperations
    }

    public fun isBiasOperation(operationName: String): Boolean {
        return operationName.lowercase() == biasOperation.lowercase()
    }

    public fun activationFor(operationName: String): MinervaActivation? {
        return activationOperations[operationName.lowercase()]
    }

    public fun layerKindFor(operationName: String): MinervaLayerKind? {
        return if (isLayerOperation(operationName)) MinervaLayerKind.DENSE else null
    }
}

/**
 * Exception raised when a validated graph still cannot be lowered into Minerva IR.
 */
public class MinervaLoweringException(
    message: String,
    public val code: String,
    public val nodeId: String? = null,
    public val operationName: String? = null,
    public val details: Map<String, String> = emptyMap()
) : IllegalArgumentException(message) {
    init {
        require(code.isNotBlank()) { "lowering exception code cannot be blank" }
    }
}

/**
 * Lowers a compatible ComputeGraph into a compact Minerva intermediate.
 */
public class MinervaGraphCanonicalizer @kotlin.jvm.JvmOverloads constructor(
    public val patternRegistry: MinervaLayerPatternRegistry = MinervaLayerPatternRegistry(),
    override val backendName: String = MinervaExportBackend.backendName
) : GraphExportConverter<ComputeGraph, MinervaIntermediate> {

    override fun convert(input: ComputeGraph, context: GraphExportContext): MinervaIntermediate {
        context.info(
            stage = GraphExportStage.LOWERING,
            code = "minerva.lowering.started",
            message = "Lowering compatible ComputeGraph to Minerva IR.",
            details = mapOf("nodes" to input.nodes.size.toString())
        )

        val topological = try {
            input.getTopologicalOrder()
        } catch (exception: Exception) {
            fail(
                context = context,
                code = "minerva.lowering.topology_invalid",
                message = exception.message ?: "Unable to determine graph topological order.",
                details = mapOf("remediation" to "Validate the graph before Minerva lowering.")
            )
        }

        val tensors = linkedMapOf<String, MinervaTensorRef>()
        val loweredNodeIds = mutableSetOf<String>()
        val layers = topological.mapNotNull { node ->
            val kind = patternRegistry.layerKindFor(node.operationName) ?: return@mapNotNull null
            lowerLayer(input, node, kind, context, tensors).also { layer ->
                loweredNodeIds += layer.sourceNodeIds
            }
        }

        if (layers.isEmpty()) {
            fail(
                context = context,
                code = "minerva.lowering.no_layers",
                message = "No lowerable Minerva layer patterns were found.",
                details = mapOf("remediation" to "Provide at least one matmul, dense, or linear layer.")
            )
        }

        val unlowered = topological.firstOrNull { node ->
            node.operationName.lowercase() != "input" && node.id !in loweredNodeIds
        }
        if (unlowered != null) {
            fail(
                context = context,
                code = "minerva.lowering.unlowered_node",
                message = "Node '${unlowered.id}' (${unlowered.operationName}) was not part of a Minerva layer pattern.",
                node = unlowered,
                details = mapOf("remediation" to "Use dense/matmul with optional add bias and activation fragments.")
            )
        }

        val projectName = context.targetName ?: "minerva_model"
        val intermediate = MinervaIntermediate(
            projectName = projectName,
            target = targetFromContext(context),
            quantization = quantizationFromContext(context),
            input = layers.first().input,
            output = layers.last().output,
            layers = layers,
            tensors = tensors.values.toList(),
            metadata = context.metadata + mapOf("lowering" to "minerva-phase-one")
        )

        context.info(
            stage = GraphExportStage.LOWERING,
            code = "minerva.lowering.completed",
            message = "Lowered ComputeGraph to Minerva IR.",
            details = mapOf(
                "projectName" to intermediate.projectName,
                "layers" to intermediate.layerCount.toString(),
                "tensors" to intermediate.tensors.size.toString(),
                "input" to intermediate.input.id,
                "output" to intermediate.output.id
            )
        )
        return intermediate
    }

    private fun lowerLayer(
        graph: ComputeGraph,
        layerNode: GraphNode,
        kind: MinervaLayerKind,
        context: GraphExportContext,
        tensors: MutableMap<String, MinervaTensorRef>
    ): MinervaLayer {
        val incoming = incomingEdges(graph, layerNode)
        if (incoming.size != 2) {
            fail(
                context = context,
                code = "minerva.lowering.layer_arity",
                message = "Layer node '${layerNode.id}' (${layerNode.operationName}) expects data and weight inputs.",
                node = layerNode,
                details = mapOf(
                    "expected" to "2",
                    "actual" to incoming.size.toString(),
                    "remediation" to "Lower dense layers as matmul(data, weight)."
                )
            )
        }

        val dataEdge = incoming[0]
        val weightEdge = incoming[1]
        val sourceNodeIds = mutableListOf(layerNode.id)
        var outputProducer = layerNode
        var outputSpec = singleOutput(layerNode, context)
        var bias: MinervaTensorRef? = null
        var activation: MinervaActivation? = null

        val firstConsumer = singleConsumerOrNull(graph, layerNode, context)
        if (firstConsumer != null && patternRegistry.isBiasOperation(firstConsumer.operationName)) {
            val addIncoming = incomingEdges(graph, firstConsumer)
            val layerToAdd = addIncoming.singleOrNull { it.source == layerNode }
                ?: fail(
                    context = context,
                    code = "minerva.lowering.bias_add_source",
                    message = "Bias add node '${firstConsumer.id}' does not consume layer '${layerNode.id}'.",
                    node = firstConsumer,
                    details = mapOf("remediation" to "Place add directly after the layer output.")
                )
            val biasEdge = addIncoming.singleOrNull { it != layerToAdd }
                ?: fail(
                    context = context,
                    code = "minerva.lowering.bias_missing",
                    message = "Bias add node '${firstConsumer.id}' does not have a separate bias input.",
                    node = firstConsumer,
                    details = mapOf("remediation" to "Provide add(layer, bias) for bias lowering.")
                )
            bias = tensorRef(
                spec = biasEdge.tensorSpec,
                role = MinervaTensorRole.BIAS,
                sourceNode = biasEdge.source,
                context = context,
                tensors = tensors
            )
            outputProducer = firstConsumer
            outputSpec = singleOutput(firstConsumer, context)
            sourceNodeIds += firstConsumer.id
        } else if (firstConsumer != null) {
            val directActivation = patternRegistry.activationFor(firstConsumer.operationName)
            if (directActivation != null) {
                activation = directActivation
                outputProducer = firstConsumer
                outputSpec = singleOutput(firstConsumer, context)
                sourceNodeIds += firstConsumer.id
            }
        }

        val activationConsumer = singleConsumerOrNull(graph, outputProducer, context)
        if (activation == null && activationConsumer != null) {
            val activationKind = patternRegistry.activationFor(activationConsumer.operationName)
            if (activationKind != null) {
                val activationIncoming = incomingEdges(graph, activationConsumer)
                if (activationIncoming.singleOrNull()?.source != outputProducer) {
                    fail(
                        context = context,
                        code = "minerva.lowering.activation_source",
                        message = "Activation node '${activationConsumer.id}' is not directly connected to the layer output.",
                        node = activationConsumer,
                        details = mapOf("remediation" to "Place activation directly after layer or bias add output.")
                    )
                }
                activation = activationKind
                outputProducer = activationConsumer
                outputSpec = singleOutput(activationConsumer, context)
                sourceNodeIds += activationConsumer.id
            }
        }

        singleConsumerOrNull(graph, outputProducer, context)

        val inputRole = if (dataEdge.source.operationName.lowercase() == "input") {
            MinervaTensorRole.INPUT
        } else {
            MinervaTensorRole.INTERMEDIATE
        }
        val outputRole = if (graph.getOutputNodes().any { it == outputProducer }) {
            MinervaTensorRole.OUTPUT
        } else {
            MinervaTensorRole.INTERMEDIATE
        }

        return MinervaLayer(
            id = layerNode.id,
            kind = kind,
            input = tensorRef(dataEdge.tensorSpec, inputRole, dataEdge.source, context, tensors),
            weights = tensorRef(weightEdge.tensorSpec, MinervaTensorRole.WEIGHT, weightEdge.source, context, tensors),
            bias = bias,
            output = tensorRef(outputSpec, outputRole, outputProducer, context, tensors),
            activation = activation,
            sourceNodeIds = sourceNodeIds,
            metadata = mapOf(
                "operationName" to layerNode.operationName,
                "operationType" to layerNode.operationType
            )
        )
    }

    private fun singleOutput(node: GraphNode, context: GraphExportContext): TensorSpec {
        if (node.outputs.size != 1) {
            fail(
                context = context,
                code = "minerva.lowering.output_arity",
                message = "Node '${node.id}' (${node.operationName}) must produce exactly one tensor.",
                node = node,
                details = mapOf("actual" to node.outputs.size.toString())
            )
        }
        return node.outputs.single()
    }

    private fun singleConsumerOrNull(
        graph: ComputeGraph,
        node: GraphNode,
        context: GraphExportContext
    ): GraphNode? {
        val consumers = outgoingEdges(graph, node).map { it.destination }
        if (consumers.size > 1) {
            fail(
                context = context,
                code = "minerva.lowering.branching",
                message = "Node '${node.id}' fans out to ${consumers.size} consumers during Minerva lowering.",
                node = node,
                details = mapOf(
                    "consumerNodeIds" to consumers.joinToString(",") { it.id },
                    "remediation" to "Use one sequential MLP chain per Minerva export."
                )
            )
        }
        return consumers.singleOrNull()
    }

    private fun incomingEdges(graph: ComputeGraph, node: GraphNode): List<GraphEdge> {
        return graph.edges
            .filter { it.destination == node }
            .sortedBy { it.destinationInputIndex }
    }

    private fun outgoingEdges(graph: ComputeGraph, node: GraphNode): List<GraphEdge> {
        return graph.edges.filter { it.source == node }
    }

    private fun tensorRef(
        spec: TensorSpec,
        role: MinervaTensorRole,
        sourceNode: GraphNode,
        context: GraphExportContext,
        tensors: MutableMap<String, MinervaTensorRef>
    ): MinervaTensorRef {
        val shape = spec.shape ?: fail(
            context = context,
            code = "minerva.lowering.dynamic_shape",
            message = "Tensor '${spec.name}' on node '${sourceNode.id}' has no static shape.",
            node = sourceNode,
            details = mapOf("remediation" to "Run Minerva compatibility validation before lowering.")
        )
        val id = tensorId(role, sourceNode.id, spec.name)
        return tensors.getOrPut(id) {
            MinervaTensorRef(
                id = id,
                name = spec.name,
                shape = shape,
                dtype = spec.dtype,
                role = role,
                sourceNodeId = sourceNode.id,
                values = tensorValues(spec, shape, sourceNode, context),
                metadata = spec.metadata.mapValues { it.value.toString() }
            )
        }
    }

    private fun tensorValues(
        spec: TensorSpec,
        shape: List<Int>,
        sourceNode: GraphNode,
        context: GraphExportContext
    ): List<Float>? {
        val elementCount = shape.fold(1) { acc, dim -> acc * dim }
        val values = when (val rawValues = spec.metadata["values"]) {
            null -> symbolicValues(spec, elementCount)
            is FloatArray -> rawValues.toList()
            is IntArray -> rawValues.map { it.toFloat() }
            is List<*> -> rawValues.map { value ->
                when (value) {
                    is Number -> value.toFloat()
                    else -> fail(
                        context = context,
                        code = "minerva.lowering.tensor_values_invalid",
                        message = "Tensor '${spec.name}' on node '${sourceNode.id}' has non-numeric initializer data.",
                        node = sourceNode,
                        details = mapOf("remediation" to "Use numeric FloatArray or IntArray initializer metadata.")
                    )
                }
            }
            else -> fail(
                context = context,
                code = "minerva.lowering.tensor_values_invalid",
                message = "Tensor '${spec.name}' on node '${sourceNode.id}' has unsupported initializer metadata.",
                node = sourceNode,
                details = mapOf(
                    "valuesType" to rawValues::class.simpleName.orEmpty(),
                    "remediation" to "Use numeric FloatArray or IntArray initializer metadata."
                )
            )
        } ?: return null
        if (values.size != elementCount) {
            fail(
                context = context,
                code = "minerva.lowering.tensor_values_shape_mismatch",
                message = "Tensor '${spec.name}' on node '${sourceNode.id}' initializer has ${values.size} value(s), expected $elementCount.",
                node = sourceNode,
                details = mapOf(
                    "actual" to values.size.toString(),
                    "expected" to elementCount.toString(),
                    "remediation" to "Match initializer data length to the tensor shape."
                )
            )
        }
        if (values.any { !it.isFinite() }) {
            fail(
                context = context,
                code = "minerva.lowering.tensor_values_non_finite",
                message = "Tensor '${spec.name}' on node '${sourceNode.id}' initializer contains non-finite values.",
                node = sourceNode,
                details = mapOf("remediation" to "Use finite numeric initializer values.")
            )
        }
        return values
    }

    private fun symbolicValues(spec: TensorSpec, elementCount: Int): List<Float>? {
        return when (val init = spec.metadata["init"]?.toString()) {
            "zeros" -> List(elementCount) { 0.0f }
            "ones" -> List(elementCount) { 1.0f }
            null, "unspecified" -> null
            else -> {
                if (init.startsWith("full(") && init.endsWith(")")) {
                    val value = spec.metadata["value"] as? Number
                        ?: init.removePrefix("full(").removeSuffix(")").toFloatOrNull()
                    if (value != null) List(elementCount) { value.toFloat() } else null
                } else {
                    null
                }
            }
        }
    }

    private fun tensorId(role: MinervaTensorRole, sourceNodeId: String, tensorName: String): String {
        val cleanName = tensorName.replace(Regex("[^A-Za-z0-9_]+"), "_").ifBlank { "tensor" }
        val cleanNode = sourceNodeId.replace(Regex("[^A-Za-z0-9_]+"), "_").ifBlank { "node" }
        return "${role.name.lowercase()}_${cleanNode}_$cleanName"
    }

    private fun targetFromContext(context: GraphExportContext): MinervaTarget {
        val compilerId = context.metadata["target"]
        return MinervaTarget.values().firstOrNull { it.compilerId == compilerId }
            ?: MinervaTarget.ATMEGA328P
    }

    private fun quantizationFromContext(context: GraphExportContext): MinervaQuantization {
        val compilerId = context.metadata["quantization"]
        return MinervaQuantization.values().firstOrNull { it.compilerId == compilerId }
            ?: MinervaQuantization.Q8
    }

    private fun fail(
        context: GraphExportContext,
        code: String,
        message: String,
        node: GraphNode? = null,
        details: Map<String, String> = emptyMap()
    ): Nothing {
        context.error(
            stage = GraphExportStage.LOWERING,
            code = code,
            message = message,
            nodeId = node?.id,
            operationName = node?.operationName,
            details = details
        )
        throw MinervaLoweringException(
            message = message,
            code = code,
            nodeId = node?.id,
            operationName = node?.operationName,
            details = details
        )
    }
}
