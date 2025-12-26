package sk.ainet.io.gguf.export

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.io.gguf.GGUF_DEFAULT_ALIGNMENT
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.exec.GraphExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.tape.ExecutionTape

/**
 * Options for preparing a GGUF export.
 *
 * @property metadataOnly If true, skip tensor payloads in the request (metadata-only export).
 * @property graphFormatVersion Version tag for graph metadata payloads.
 */
public data class GgufExportOptions(
    val metadataOnly: Boolean = false,
    val graphFormatVersion: Int = 1,
    val defaultDtype: String? = "FP32",
    val generalMetadata: Map<String, Any> = emptyMap(),
    val provenance: Map<String, Any> = emptyMap()
)

/** Tensor entry to be consumed by a future GGUF writer implementation. */
public data class GgufTensorEntry(
    val ggufName: String,
    val tensor: Tensor<*, *>,
    val quantization: GGMLQuantizationType,
    val shape: List<Int>
)

/** Aggregate export payload prepared by the facade; writer will consume this. */
public data class GgufWriteRequest(
    val metadata: Map<String, Any>,
    val tensors: List<GgufTensorEntry>,
    val tensorMap: Map<String, String>
)

/**
 * Prepare a GGUF write request from a compute graph and weight tensors.
 * This does not write bytes; a future GGUFWriter will consume the request.
 */
public fun exportGraphToGguf(
    graph: ComputeGraph,
    weights: Map<String, Tensor<*, *>>,
    label: String = "graph",
    options: GgufExportOptions = GgufExportOptions()
): GgufWriteRequest {
    validateGraph(graph)
    validateWeights(weights)
    val tensorMap: Map<String, String> = weights.keys.sorted().associateWith { it }
    val entries: List<GgufTensorEntry> = if (options.metadataOnly) {
        emptyList()
    } else {
        weights.entries
            .sortedBy { it.key }
            .map { entry ->
                val name = entry.key
                val tensor = entry.value
                GgufTensorEntry(
                    ggufName = name,
                    tensor = tensor,
                    quantization = inferQuantization(tensor),
                    shape = tensor.shape.dimensions.toList()
                )
            }
    }

    val metadata = buildGraphMetadata(graph, label, tensorMap, options)

    return GgufWriteRequest(
        metadata = metadata,
        tensors = entries,
        tensorMap = tensorMap
    )
}

/**
 * Build a GGUF export request starting from an ExecutionTape (if present).
 * Falls back to an empty graph when no tape is supplied.
 */
public fun exportTapeToGguf(
    tape: ExecutionTape?,
    weights: Map<String, Tensor<*, *>>,
    label: String = "graph",
    options: GgufExportOptions = GgufExportOptions()
): GgufWriteRequest {
    val graph = when (tape) {
        is DefaultExecutionTape -> tape.toComputeGraph()
        else -> DefaultComputeGraph()
    }
    return exportGraphToGguf(graph, weights, label, options)
}

/**
 * Convenience: record a forward pass under a graph/tape context, collect weights, and prepare a GGUF request.
 * The forwardPass lambda receives a GraphExecutionContext whose ops are tracing-enabled (VoidTensorOps base).
 */
public fun exportModelToGguf(
    model: Module<*, *>,
    forwardPass: (GraphExecutionContext) -> Unit,
    label: String = model.name,
    options: GgufExportOptions = GgufExportOptions(),
    baseOps: TensorOps = VoidTensorOps()
): GgufWriteRequest {
    val graph = DefaultComputeGraph()
    val ctx = DefaultGraphExecutionContext.tapeAndGraph(baseOps = baseOps, graph = graph)
    ctx.startRecording()
    forwardPass(ctx)
    val tape = ctx.stopRecording() as? DefaultExecutionTape
    val builtGraph = tape?.toComputeGraph() ?: graph
    val weights = collectParameters(model)
    return exportGraphToGguf(builtGraph, weights, label, options)
}

/**
 * Recursively collect parameters from a Module tree using a stable path-based naming scheme.
 */
public fun collectParameters(model: Module<*, *>, prefix: String = model.name): Map<String, Tensor<*, *>> {
    val result = linkedMapOf<String, Tensor<*, *>>()
    if (model is ModuleParameters<*, *>) {
        @Suppress("UNCHECKED_CAST")
        val params = (model as ModuleParameters<*, *>).params
        params.forEach { p ->
            val key = if (prefix.isBlank()) p.name else "$prefix.${p.name}"
            @Suppress("UNCHECKED_CAST")
            result[key] = p.value as Tensor<*, *>
        }
    }
    model.modules.forEachIndexed { idx, child ->
        val childPrefix = if (child.name.isNotBlank()) "$prefix.${child.name}" else "$prefix.$idx"
        result.putAll(collectParameters(child, childPrefix))
    }
    return result
}

/** Convenience: prepare and write GGUF bytes directly for a graph + weights. */
public fun writeGraphToGgufBytes(
    graph: ComputeGraph,
    weights: Map<String, Tensor<*, *>>,
    label: String = "graph",
    options: GgufExportOptions = GgufExportOptions()
): Pair<GGUFWriteReport, ByteArray> {
    val request = exportGraphToGguf(graph, weights, label, options)
    return GGUFWriter.writeToByteArray(request)
}

/** Convenience: prepare and write GGUF bytes directly for a model + forward pass. */
public fun writeModelToGgufBytes(
    model: Module<*, *>,
    forwardPass: (GraphExecutionContext) -> Unit,
    label: String = model.name,
    options: GgufExportOptions = GgufExportOptions(),
    baseOps: TensorOps = VoidTensorOps()
): Pair<GGUFWriteReport, ByteArray> {
    val request = exportModelToGguf(model, forwardPass, label, options, baseOps)
    return GGUFWriter.writeToByteArray(request)
}

private fun inferQuantization(tensor: Tensor<*, *>): GGMLQuantizationType {
    val dtype = tensor.dtype
    return when (dtype) {
        FP32::class -> GGMLQuantizationType.F32
        FP16::class -> GGMLQuantizationType.F16
        Int8::class -> GGMLQuantizationType.I8
        Int32::class -> GGMLQuantizationType.I32
        else -> GGMLQuantizationType.F32
    }
}

private fun buildGraphMetadata(
    graph: ComputeGraph,
    label: String,
    tensorMap: Map<String, String>,
    options: GgufExportOptions
): Map<String, Any> {
    val nodesJson = encodeNodes(graph.nodes)
    val edgesJson = encodeEdges(graph.edges)
    val tensorMapJson = encodeTensorMap(tensorMap)
    val out = linkedMapOf<String, Any>(
        "model.name" to label,
        "general.alignment" to GGUF_DEFAULT_ALIGNMENT,
        "skainet.graph.format_version" to options.graphFormatVersion,
        "skainet.graph.nodes" to nodesJson,
        "skainet.graph.edges" to edgesJson,
        "skainet.tensor.map" to tensorMapJson,
        "skainet.tensor.count" to tensorMap.size
    )
    options.defaultDtype?.let { out["skainet.dtype.default"] = it }
    options.generalMetadata.entries.sortedBy { it.key }.forEach { (k, v) ->
        out[k] = v
    }
    options.provenance.entries.sortedBy { it.key }.forEach { (k, v) ->
        out[k] = v
    }
    return out
}

private fun encodeNodes(nodes: List<GraphNode>): String {
    val ordered = nodes.sortedBy { it.id }
    val sb = StringBuilder()
    sb.append('[')
    ordered.forEachIndexed { idx, node ->
        if (idx > 0) sb.append(',')
        sb.append('{')
        sb.append("\"id\":\"").append(node.id.escapeJson()).append("\",")
        sb.append("\"op\":\"").append(node.operationName.escapeJson()).append("\",")
        sb.append("\"type\":\"").append(node.operation.type.escapeJson()).append("\",")
        sb.append("\"inputs\":").append(encodeSpecs(node.inputs)).append(',')
        sb.append("\"outputs\":").append(encodeSpecs(node.outputs)).append(',')
        sb.append("\"params\":").append(encodeParameters(node.operation.parameters))
        sb.append('}')
    }
    sb.append(']')
    return sb.toString()
}

private fun encodeEdges(edges: List<GraphEdge>): String {
    val ordered = edges.sortedBy { it.id }
    val sb = StringBuilder()
    sb.append('[')
    ordered.forEachIndexed { idx, edge ->
        if (idx > 0) sb.append(',')
        sb.append('{')
        sb.append("\"id\":\"").append(edge.id.escapeJson()).append("\",")
        sb.append("\"src\":\"").append(edge.source.id.escapeJson()).append("\",")
        sb.append("\"dst\":\"").append(edge.destination.id.escapeJson()).append("\",")
        sb.append("\"srcOut\":").append(edge.sourceOutputIndex).append(',')
        sb.append("\"dstIn\":").append(edge.destinationInputIndex).append(',')
        sb.append("\"tensor\":\"").append(edge.tensorSpec.name.escapeJson()).append("\"")
        sb.append('}')
    }
    sb.append(']')
    return sb.toString()
}

private fun encodeSpecs(specs: List<TensorSpec>): String {
    val sb = StringBuilder()
    sb.append('[')
    specs.forEachIndexed { idx, spec ->
        if (idx > 0) sb.append(',')
        sb.append('{')
        sb.append("\"name\":\"").append(spec.name.escapeJson()).append("\",")
        sb.append("\"dtype\":\"").append(spec.dtype.escapeJson()).append("\",")
        sb.append("\"shape\":").append(spec.shape?.let { encodeShape(it) } ?: "null")
        sb.append('}')
    }
    sb.append(']')
    return sb.toString()
}

private fun encodeShape(shape: List<Int>): String {
    val sb = StringBuilder()
    sb.append('[')
    shape.forEachIndexed { idx, dim ->
        if (idx > 0) sb.append(',')
        sb.append(dim)
    }
    sb.append(']')
    return sb.toString()
}

private fun encodeTensorMap(map: Map<String, String>): String {
    val ordered = map.entries.sortedBy { it.key }
    val sb = StringBuilder()
    sb.append('{')
    ordered.forEachIndexed { idx, entry ->
        if (idx > 0) sb.append(',')
        sb.append("\"").append(entry.key.escapeJson()).append("\":\"")
            .append(entry.value.escapeJson()).append("\"")
    }
    sb.append('}')
    return sb.toString()
}

private fun encodeParameters(params: Map<String, Any>): String {
    if (params.isEmpty()) return "{}"
    val keys = params.keys.sorted()
    val sb = StringBuilder()
    sb.append('{')
    keys.forEachIndexed { idx, k ->
        if (idx > 0) sb.append(',')
        sb.append("\"").append(k.escapeJson()).append("\":")
        val v = params[k]
        sb.append(encodeValue(v))
    }
    sb.append('}')
    return sb.toString()
}

private fun encodeValue(value: Any?): String = when (value) {
    null -> "null"
    is Number, is Boolean -> value.toString()
    is String -> "\"${value.escapeJson()}\""
    is List<*> -> encodeList(value)
    is Map<*, *> -> encodeMap(value)
    else -> "\"${value.toString().escapeJson()}\""
}

private fun encodeList(values: List<*>): String {
    val sb = StringBuilder()
    sb.append('[')
    values.forEachIndexed { idx, v ->
        if (idx > 0) sb.append(',')
        sb.append(encodeValue(v))
    }
    sb.append(']')
    return sb.toString()
}

private fun encodeMap(values: Map<*, *>): String {
    val ordered = values.entries.sortedBy { it.key.toString() }
    val sb = StringBuilder()
    sb.append('{')
    ordered.forEachIndexed { idx, entry ->
        if (idx > 0) sb.append(',')
        val key = entry.key?.toString() ?: "null"
        sb.append("\"").append(key.escapeJson()).append("\":")
        sb.append(encodeValue(entry.value))
    }
    sb.append('}')
    return sb.toString()
}

private fun String.escapeJson(): String = buildString(length) {
    for (ch in this@escapeJson) {
        when (ch) {
            '\\' -> append("\\\\")
            '"' -> append("\\\"")
            '\b' -> append("\\b")
            '\u000C' -> append("\\f")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> append(ch)
        }
    }
}

private fun validateGraph(graph: ComputeGraph) {
    when (val result = graph.validate()) {
        ValidationResult.Valid -> Unit
        is ValidationResult.Invalid -> error("Graph validation failed: ${result.errors.joinToString("; ")}")
    }
}

private fun validateWeights(weights: Map<String, Tensor<*, *>>) {
    val duplicateKeys = weights.keys.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
    require(duplicateKeys.isEmpty()) { "Duplicate weight keys: $duplicateKeys" }
    weights.forEach { (name, tensor) ->
        val dims = tensor.shape.dimensions
        require(dims.all { it > 0 }) {
            "Tensor $name has non-positive dimensions ${dims.toList()}"
        }
    }
}
