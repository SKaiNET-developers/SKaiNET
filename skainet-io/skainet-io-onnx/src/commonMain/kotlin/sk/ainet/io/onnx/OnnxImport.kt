package sk.ainet.io.onnx

import onnx.AttributeProto
import onnx.ModelProto
import onnx.TensorProto
import onnx.ValueInfoProto
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.Conv2dOperation
import sk.ainet.lang.tensor.ops.DivideOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.MaxPool2dOperation
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.ReshapeOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.SigmoidOperation
import sk.ainet.lang.tensor.ops.SoftmaxOperation
import sk.ainet.lang.tensor.ops.SubtractOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.TransposeOperation
import sk.ainet.lang.tensor.ops.Upsample2dOperation
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType

/**
 * Minimal, proto-agnostic view of an ONNX graph.
 * This keeps the importer independent of the generated proto surface while
 * we wire in pbandk-generated classes.
 */
public data class OnnxGraphView(
    val nodes: List<OnnxNodeView>,
    val initializers: Map<String, OnnxTensorView>,
    val inputs: List<OnnxValueInfoView>,
    val outputs: List<OnnxValueInfoView>,
    val valueInfos: Map<String, OnnxValueInfoView>
)

public data class OnnxNodeView(
    val name: String,
    val opType: String,
    val inputs: List<String>,
    val outputs: List<String>,
    val attributes: Map<String, OnnxAttributeValue>
)

public data class OnnxTensorView(
    val name: String,
    val shape: List<Long>,
    val dtype: String,
    val proto: TensorProto
)

public data class OnnxValueInfoView(
    val name: String,
    val shape: List<Int>?,
    val dtype: String?
)

public data class OnnxAttributeValue(
    val name: String,
    val ints: List<Long> = emptyList(),
    val floats: List<Float> = emptyList(),
    val strings: List<String> = emptyList(),
    val tensor: TensorProto? = null
)

/**
 * Stub importer that will map an ONNX graph view to a ComputeGraph.
 * The mapping logic will be filled in as operator support lands.
 */
public class OnnxToComputeGraphImporter {

    private data class ProducedTensor(
        val node: GraphNode,
        val outputIndex: Int,
        val tensorSpec: TensorSpec
    )

    public fun import(graph: OnnxGraphView): ComputeGraph {
        val computeGraph = DefaultComputeGraph()
        val producers = mutableMapOf<String, ProducedTensor>()

        val tensorInfo = buildTensorInfo(graph)

        // Add initializer-backed constant nodes first so downstream edges can connect.
        graph.initializers.values.forEach { initializer ->
            val spec = initializer.toTensorSpec()
            val node = GraphNode(
                id = "init_${initializer.name}",
                operation = InitializerOperation(initializer.name),
                inputs = emptyList(),
                outputs = listOf(spec),
                metadata = mapOf("initializer" to initializer.proto)
            )
            computeGraph.addNode(node)
            producers[initializer.name] = ProducedTensor(node, 0, spec)
        }

        // Add declared graph inputs (excluding those already covered by initializers).
        graph.inputs.forEachIndexed { idx, input ->
            if (producers.containsKey(input.name)) return@forEachIndexed
            val spec = input.toTensorSpec()
            val node = GraphNode(
                id = if (input.name.isNotEmpty()) input.name else "input_$idx",
                operation = InputOperation<DType, Any>(),
                inputs = emptyList(),
                outputs = listOf(spec),
                metadata = mapOf("onnxInput" to true)
            )
            computeGraph.addNode(node)
            producers[input.name] = ProducedTensor(node, 0, spec)
        }

        // Map each ONNX node to a ComputeGraph node and wire edges.
        graph.nodes.forEach { nodeView ->
            val op = nodeView.toOperation(graph.initializers)

            // Ops like Clip/Pad fold their trailing initializer inputs (min/max, pads) into
            // parameters at import time — only the data input stays wired.
            val nodeInputs = if (op.parameters[ONNX_INPUTS_CONSUMED_AT_IMPORT] == true) {
                nodeView.inputs.take(1)
            } else {
                nodeView.inputs
            }

            val inputSpecs = nodeInputs.mapNotNull { inputName ->
                if (inputName.isEmpty()) return@mapNotNull null
                val produced = ensureProducer(inputName, tensorInfo, computeGraph, producers)
                produced.tensorSpec
            }

            val outputSpecs = nodeView.outputs.mapIndexed { idx, outputName ->
                resolveTensorSpec(outputName, tensorInfo, inputSpecs, op, idx)
            }

            val initMetadata = graph.initializers.filterKeys { it in nodeInputs }
            val graphNode = GraphNode(
                id = nodeView.name,
                operation = op,
                inputs = inputSpecs,
                outputs = outputSpecs,
                metadata = buildMap<String, Any> {
                    put("attributes", nodeView.attributes)
                    if (initMetadata.isNotEmpty()) put("initializers", initMetadata)
                }
            )
            computeGraph.addNode(graphNode)

            // Wire edges now that we have a concrete destination.
            nodeInputs.forEachIndexed { inputIdx, inputName ->
                if (inputName.isEmpty()) return@forEachIndexed
                val produced = ensureProducer(inputName, tensorInfo, computeGraph, producers)
                computeGraph.addEdge(
                    GraphEdge(
                        id = "${produced.node.id}->${graphNode.id}[$inputIdx]",
                        source = produced.node,
                        destination = graphNode,
                        sourceOutputIndex = produced.outputIndex,
                        destinationInputIndex = inputIdx,
                        tensorSpec = produced.tensorSpec
                    )
                )
            }

            // Register outputs as producers for downstream consumers.
            outputSpecs.forEachIndexed { idx, spec ->
                producers[spec.name] = ProducedTensor(graphNode, idx, spec)
            }
        }

        return computeGraph
    }

    private fun ensureProducer(
        inputName: String,
        tensorInfo: Map<String, OnnxValueInfoView>,
        computeGraph: DefaultComputeGraph,
        producers: MutableMap<String, ProducedTensor>
    ): ProducedTensor {
        producers[inputName]?.let { return it }
        val spec = tensorInfo[inputName]?.toTensorSpec()
            ?: TensorSpec(name = inputName, shape = null, dtype = "unknown")
        val node = GraphNode(
            id = "implicit_input_$inputName",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(spec),
            metadata = mapOf("onnxImplicitInput" to true)
        )
        computeGraph.addNode(node)
        val produced = ProducedTensor(node, 0, spec)
        producers[inputName] = produced
        return produced
    }

    private fun buildTensorInfo(graph: OnnxGraphView): Map<String, OnnxValueInfoView> {
        val fromValueInfo = graph.valueInfos
        val fromInputs = graph.inputs.associateBy { it.name }
        val fromOutputs = graph.outputs.associateBy { it.name }
        return buildMap {
            putAll(fromValueInfo)
            putAll(fromInputs)
            putAll(fromOutputs)
        }
    }

    private fun resolveTensorSpec(
        outputName: String,
        tensorInfo: Map<String, OnnxValueInfoView>,
        inputSpecs: List<TensorSpec>,
        op: Operation,
        outputIndex: Int
    ): TensorSpec {
        val info = tensorInfo[outputName]
        val dtype = info?.dtype
            ?: inputSpecs.firstOrNull()?.dtype
            ?: op.parameters["dtype"] as? String
            ?: "unknown"
        val shape = info?.shape
            ?: opShapeInference(op, inputSpecs, outputIndex)
            ?: broadcastShape(inputSpecs.map { it.shape })
        return TensorSpec(
            name = if (outputName.isNotEmpty()) outputName else "${op.name}_output_$outputIndex",
            shape = shape,
            dtype = dtype
        )
    }

    private fun OnnxNodeView.toOperation(initializers: Map<String, OnnxTensorView>): Operation {
        val upper = opType.uppercase()
        val attrInts = { key: String -> attributes[key]?.ints?.map { it.toSafeInt() } }
        return when (upper) {
            "CONV" -> Conv2dOperation<DType, Any>(
                parameters = buildMap {
                    attrInts("strides")?.let { put("strides", it) }
                    attrInts("pads")?.let { put("pads", it) }
                    attrInts("dilations")?.let { put("dilations", it) }
                    attributes["group"]?.ints?.firstOrNull()?.toSafeInt()?.let { put("groups", it) }
                    attrInts("kernel_shape")?.let { put("kernelShape", it) }
                }
            )
            "RELU" -> ReluOperation<DType, Any>()
            "SIGMOID" -> SigmoidOperation<DType, Any>()
            "LEAKYRELU" -> OnnxMappedOperation(
                opName = "leakyRelu",
                type = "activation",
                // ONNX attribute `alpha` (default 0.01) is SKaiNET's negativeSlope.
                parameters = mapOf("negativeSlope" to (attributes["alpha"]?.floats?.firstOrNull() ?: 0.01f))
            )
            "SILU" -> OnnxMappedOperation(opName = "silu", type = "activation")
            "CLIP" -> {
                // Opset < 11 carries min/max as attributes; opset >= 11 as optional initializer
                // inputs. A min/max fed by a *dynamic* tensor cannot fold to a clamp — keep the
                // placeholder so execution fails loudly instead of silently not clipping.
                val minResolved = attributes["min"]?.floats?.firstOrNull()
                    ?: resolveScalarFloatInitializer(inputs.getOrNull(1), initializers)
                val maxResolved = attributes["max"]?.floats?.firstOrNull()
                    ?: resolveScalarFloatInitializer(inputs.getOrNull(2), initializers)
                val hasDynamicBound = listOfNotNull(inputs.getOrNull(1), inputs.getOrNull(2))
                    .any { it.isNotEmpty() && it !in initializers }
                if (hasDynamicBound) {
                    OnnxPlaceholderOperation(
                        opType = opType,
                        type = "onnx",
                        parameters = mapOf("reason" to "Clip with dynamic min/max inputs is not supported")
                    )
                } else {
                    OnnxMappedOperation(
                        opName = "clamp",
                        type = "activation",
                        parameters = mapOf(
                            "minVal" to (minResolved ?: -Float.MAX_VALUE),
                            "maxVal" to (maxResolved ?: Float.MAX_VALUE),
                            ONNX_INPUTS_CONSUMED_AT_IMPORT to true,
                        )
                    )
                }
            }
            "PAD" -> {
                // Only the form pad2d covers: constant mode, zero fill, 4D input with no
                // batch/channel padding. pads order (4D): [Nb, Cb, Hb, Wb, Ne, Ce, He, We].
                val mode = attributes["mode"]?.strings?.firstOrNull() ?: "constant"
                val pads = attrInts("pads")
                    ?: inputs.getOrNull(1)?.takeIf { it.isNotEmpty() }?.let { initializers[it]?.intListOrNull() }
                val constantValue = attributes["value"]?.floats?.firstOrNull()
                    ?: resolveScalarFloatInitializer(inputs.getOrNull(2), initializers)
                    ?: 0f
                if (mode == "constant" && constantValue == 0f &&
                    pads != null && pads.size == 8 &&
                    pads[0] == 0 && pads[1] == 0 && pads[4] == 0 && pads[5] == 0
                ) {
                    OnnxPadOperation(padTop = pads[2], padLeft = pads[3], padBottom = pads[6], padRight = pads[7])
                } else {
                    OnnxPlaceholderOperation(
                        opType = opType,
                        type = "onnx",
                        parameters = mapOf(
                            "reason" to "Pad supports constant mode with zero value and 4D spatial-only pads",
                            "mode" to mode,
                            "pads" to (pads ?: emptyList<Int>()),
                        )
                    )
                }
            }
            "ADD" -> AddOperation<DType, Any>()
            "MUL" -> MultiplyOperation<DType, Any>()
            "DIV" -> DivideOperation<DType, Any>()
            "SUB" -> SubtractOperation<DType, Any>()
            "MATMUL" -> MatmulOperation<DType, Any>()
            "GEMM" -> MatmulOperation<DType, Any>(parameters = mapOf("onnxGemm" to true))
            "MAXPOOL" -> MaxPool2dOperation<DType, Any>(
                parameters = attrInts("kernel_shape")?.let { mapOf("kernelShape" to it) } ?: emptyMap()
            )
            "UPSAMPLE", "RESIZE" -> Upsample2dOperation<DType, Any>(
                parameters = attrInts("scales")?.let { mapOf("scale" to it) } ?: emptyMap()
            )
            "RESHAPE" -> ReshapeOperation<DType, Any>(
                parameters = attrInts("shape")?.let { mapOf("newShape" to it) } ?: emptyMap()
            )
            "TRANSPOSE" -> TransposeOperation<DType, Any>(
                parameters = attrInts("perm")?.let { mapOf("perm" to it) } ?: emptyMap()
            )
            "BATCHNORMALIZATION" -> OnnxMappedOperation(
                opName = "batchNorm",
                type = "normalization",
                // Inference-mode BN: inputs are (X, scale, B, mean, var); only epsilon matters
                // (momentum/spatial are training-time attributes).
                parameters = mapOf("epsilon" to (attributes["epsilon"]?.floats?.firstOrNull() ?: 1e-5f))
            )
            "CONCAT" -> OnnxConcatOperation(
                axis = attributes["axis"]?.ints?.firstOrNull()?.toSafeInt() ?: 0
            )
            "SLICE" -> OnnxSliceOperation(
                starts = attributes["starts"]?.ints?.map { it.toSafeInt() }.orEmpty(),
                ends = attributes["ends"]?.ints?.map { it.toSafeInt() }.orEmpty(),
                axes = attributes["axes"]?.ints?.map { it.toSafeInt() },
                steps = attributes["steps"]?.ints?.map { it.toSafeInt() }
            )
            "SPLIT" -> OnnxSplitOperation(
                axis = attributes["axis"]?.ints?.firstOrNull()?.toSafeInt() ?: 0,
                split = attributes["split"]?.ints?.map { it.toSafeInt() }
            )
            "SOFTMAX" -> SoftmaxOperation<DType, Any>()
            else -> OnnxPlaceholderOperation(
                opType = opType,
                type = "onnx",
                parameters = attributes.mapValues { (_, v) -> v.ints.ifEmpty { v.floats.ifEmpty { v.strings } } }
            )
        }
    }
}

/**
 * Utility to build a simplified graph view from a parsed ONNX model.
 */
public fun ModelProto.toGraphView(): OnnxGraphView {
    val g = graph ?: error("ONNX model is missing GraphProto")
    val initializerViews = g.initializer.associateBy { it.name }.mapValues { (name, tensor) ->
        OnnxTensorView(
            name = name,
            shape = tensor.dims,
            dtype = onnx.TensorProto.DataType.fromValue(tensor.dataType).name ?: tensor.dataType.toString(),
            proto = tensor
        )
    }
    val inputs = g.input.map { it.toView() }
    val outputs = g.output.map { it.toView() }
    val valueInfos = g.valueInfo.associateBy { it.name }.mapValues { it.value.toView() }
    val nodeViews = g.node.mapIndexed { idx, node ->
        OnnxNodeView(
            name = node.name.ifEmpty { "node_${idx}_${node.opType}" },
            opType = node.opType,
            inputs = node.input,
            outputs = node.output,
            attributes = node.attribute.associateBy { it.name }.mapValues { (_, attr) -> attr.toValue() }
        )
    }
    return OnnxGraphView(
        nodes = nodeViews,
        initializers = initializerViews,
        inputs = inputs,
        outputs = outputs,
        valueInfos = valueInfos
    )
}

private fun TensorProto.toTensorSpec(): TensorSpec = TensorSpec(
    name = name,
    shape = dims.map { it.toSafeInt() },
    dtype = TensorProto.DataType.fromValue(dataType).name ?: dataType.toString(),
    metadata = mapOf("initializer" to true)
)

private fun OnnxTensorView.toTensorSpec(): TensorSpec = TensorSpec(
    name = name,
    shape = shape.map { it.toSafeInt() },
    dtype = dtype,
    metadata = mapOf("initializer" to true)
)

private fun OnnxValueInfoView.toTensorSpec(): TensorSpec = TensorSpec(
    name = name,
    shape = shape,
    dtype = dtype ?: "unknown"
)

private fun opShapeInference(op: Operation, inputs: List<TensorSpec>, outputIndex: Int): List<Int>? {
    return when (op) {
        is OnnxConcatOperation -> op.inferOutputs(inputs).getOrNull(outputIndex)?.shape
        is OnnxSliceOperation -> op.inferOutputs(inputs).getOrNull(outputIndex)?.shape
        is OnnxSplitOperation -> op.inferOutputs(inputs).getOrNull(outputIndex)?.shape
        is OnnxPadOperation -> op.inferOutputs(inputs).getOrNull(outputIndex)?.shape
        is Conv2dOperation<*, *> -> inputs.firstOrNull()?.shape
        is MaxPool2dOperation<*, *> -> inputs.firstOrNull()?.shape
        is Upsample2dOperation<*, *> -> op.inferOutputs(inputs).getOrNull(outputIndex)?.shape
        is ReshapeOperation<*, *> -> (op.parameters["newShape"] as? List<*>)?.mapNotNull { (it as? Number)?.toInt() }
        is TransposeOperation<*, *> -> inputs.firstOrNull()?.shape?.reversed()
        else -> null
    }
}

private fun broadcastShape(shapes: List<List<Int>?>): List<Int>? {
    val present = shapes.filterNotNull()
    if (present.isEmpty()) return null
    val maxRank = present.maxOf { it.size }
    val padded = present.map { List(maxRank - it.size) { 1 } + it }
    val result = MutableList(maxRank) { 1 }
    for (dim in 0 until maxRank) {
        val dimsAt = padded.map { it[dim] }
        val target = dimsAt.maxOrNull() ?: 1
        if (dimsAt.any { it != 1 && it != target }) return null
        result[dim] = target
    }
    return result
}

private fun AttributeProto.toValue(): OnnxAttributeValue = OnnxAttributeValue(
    name = name,
    ints = when {
        ints.isNotEmpty() -> ints
        i != 0L -> listOf(i)
        else -> emptyList()
    },
    floats = when {
        floats.isNotEmpty() -> floats
        f != 0f -> listOf(f)
        else -> emptyList()
    },
    strings = if (s.array.isNotEmpty()) listOf(s.array.decodeToString()) else strings.map { it.array.decodeToString() },
    tensor = if (t != null && t != TensorProto.defaultInstance) t else null
)

private fun ValueInfoProto.toView(): OnnxValueInfoView = OnnxValueInfoView(
    name = name,
    shape = type?.tensorType?.shape?.dim?.mapNotNull { dim ->
        dim.dimValue?.toSafeInt()
    },
    dtype = type?.tensorType?.elemType?.let { TensorProto.DataType.fromValue(it).name }
)

private fun Long.toSafeInt(): Int = when {
    this > Int.MAX_VALUE -> Int.MAX_VALUE
    this < Int.MIN_VALUE -> Int.MIN_VALUE
    else -> toInt()
}

/**
 * Marks an operation whose trailing ONNX inputs (e.g. Clip's min/max, Pad's pads tensor) were
 * folded into [Operation.parameters] at import time: only the first (data) input is wired.
 */
internal const val ONNX_INPUTS_CONSUMED_AT_IMPORT: String = "onnxInputsConsumedAtImport"

/** Read a scalar FLOAT initializer (float_data or little-endian raw_data), or null. */
private fun resolveScalarFloatInitializer(
    inputName: String?,
    initializers: Map<String, OnnxTensorView>
): Float? {
    if (inputName.isNullOrEmpty()) return null
    val proto = initializers[inputName]?.proto ?: return null
    proto.floatData.firstOrNull()?.let { return it }
    val raw = proto.rawData.array
    if (proto.dataType == TensorProto.DataType.FLOAT.value && raw.size >= 4) {
        val bits = (raw[0].toInt() and 0xFF) or
            ((raw[1].toInt() and 0xFF) shl 8) or
            ((raw[2].toInt() and 0xFF) shl 16) or
            ((raw[3].toInt() and 0xFF) shl 24)
        return Float.fromBits(bits)
    }
    return null
}

/** Read an INT64 initializer as an Int list (int64_data or little-endian raw_data), or null. */
private fun OnnxTensorView.intListOrNull(): List<Int>? {
    if (proto.int64Data.isNotEmpty()) return proto.int64Data.map { it.toSafeInt() }
    val raw = proto.rawData.array
    if (proto.dataType == TensorProto.DataType.INT64.value && raw.isNotEmpty() && raw.size % 8 == 0) {
        return List(raw.size / 8) { i ->
            var v = 0L
            for (b in 7 downTo 0) {
                v = (v shl 8) or (raw[i * 8 + b].toLong() and 0xFF)
            }
            v.toSafeInt()
        }
    }
    return null
}

private class InitializerOperation(
    tensorName: String
) : BasePlaceholderOperation("onnx_initializer", mapOf("name" to tensorName))

/**
 * An ONNX node mapped to a real SKaiNET operation: the name matches the ComputeGraphExecutor
 * dispatch name and [parameters] carries the attributes it reads. `execute` is never called
 * directly — the executor dispatches on the operation name, the same way OnnxConcatOperation
 * and friends run.
 */
private class OnnxMappedOperation(
    opName: String,
    type: String,
    parameters: Map<String, Any> = emptyMap(),
) : BasePlaceholderOperation(opName, parameters, type)

/**
 * ONNX `Pad` folded to SKaiNET `pad2d` (constant mode, zero fill, 4D, spatial-only pads).
 * Carries its own shape inference: the output grows by the pads on H and W.
 */
private class OnnxPadOperation(
    private val padTop: Int,
    private val padLeft: Int,
    private val padBottom: Int,
    private val padRight: Int,
) : BasePlaceholderOperation(
    "pad2d",
    mapOf(
        "padTop" to padTop,
        "padLeft" to padLeft,
        "padBottom" to padBottom,
        "padRight" to padRight,
        ONNX_INPUTS_CONSUMED_AT_IMPORT to true,
    ),
    "shape",
) {
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        val inputShape = inputs.firstOrNull()?.shape
        val outputShape = if (inputShape != null && inputShape.size == 4) {
            listOf(
                inputShape[0],
                inputShape[1],
                inputShape[2] + padTop + padBottom,
                inputShape[3] + padLeft + padRight,
            )
        } else {
            inputShape
        }
        return listOf(
            TensorSpec(
                name = "pad2d_output",
                shape = outputShape,
                dtype = inputs.firstOrNull()?.dtype ?: "unknown"
            )
        )
    }
}

private class OnnxPlaceholderOperation(
    opType: String,
    type: String = "onnx",
    parameters: Map<String, Any?> = emptyMap()
) : BasePlaceholderOperation(
    opType,
    parameters.entries.mapNotNull { (k, v) -> v?.let { k to it } }.toMap(),
    type
)

private class OnnxConcatOperation(
    private val axis: Int
) : BasePlaceholderOperation("concat", mapOf("axis" to axis), "shape") {
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        val baseShape = inputs.firstOrNull()?.shape
        val rank = baseShape?.size ?: return listOf(
            TensorSpec(name = "concat_output", shape = null, dtype = inputs.firstOrNull()?.dtype ?: "unknown")
        )
        val outDims = baseShape.toMutableList()
        inputs.drop(1).forEach { spec ->
            val shape = spec.shape ?: return@forEach
            if (shape.size == rank && axis in shape.indices && axis in outDims.indices) {
                outDims[axis] = outDims[axis] + shape[axis]
            }
        }
        return listOf(
            TensorSpec(
                name = "concat_output",
                shape = outDims,
                dtype = inputs.firstOrNull()?.dtype ?: "unknown"
            )
        )
    }
}

private class OnnxSliceOperation(
    private val starts: List<Int>,
    private val ends: List<Int>,
    private val axes: List<Int>?,
    private val steps: List<Int>?
) : BasePlaceholderOperation(
    "slice",
    buildMap<String, Any> {
        put("starts", starts)
        put("ends", ends)
        axes?.let { put("axes", it) }
        steps?.let { put("steps", it) }
    },
    "shape"
) {
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        val inputShape = inputs.firstOrNull()?.shape ?: return listOf(
            TensorSpec(name = "slice_output", shape = null, dtype = inputs.firstOrNull()?.dtype ?: "unknown")
        )
        if (starts.isEmpty() || ends.isEmpty()) return listOf(
            TensorSpec(name = "slice_output", shape = inputShape, dtype = inputs.firstOrNull()?.dtype ?: "unknown")
        )
        val axesList = axes ?: (starts.indices).toList()
        val outDims = inputShape.toMutableList()
        axesList.forEachIndexed { idx, axis ->
            if (axis in outDims.indices) {
                val step = steps?.getOrNull(idx) ?: 1
                val length = ((ends.getOrNull(idx) ?: outDims[axis]) - (starts.getOrNull(idx)
                    ?: 0)) / step
                outDims[axis] = length.coerceAtLeast(0)
            }
        }
        return listOf(
            TensorSpec(
                name = "slice_output",
                shape = outDims,
                dtype = inputs.firstOrNull()?.dtype ?: "unknown"
            )
        )
    }
}

private class OnnxSplitOperation(
    private val axis: Int,
    private val split: List<Int>?
) : BasePlaceholderOperation(
    "split",
    buildMap<String, Any> {
        put("axis", axis)
        split?.let { put("split", it) }
    },
    "shape"
) {
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        val input = inputs.firstOrNull() ?: return emptyList()
        val inputShape = input.shape
        if (inputShape == null) return listOf(
            TensorSpec(name = "split_output_0", shape = null, dtype = input.dtype)
        )
        val sizes = split ?: run {
            val parts = 2
            val dim = inputShape.getOrNull(axis)?.coerceAtLeast(0) ?: 0
            List(parts) { dim / parts }
        }
        val outputs = mutableListOf<TensorSpec>()
        sizes.forEachIndexed { idx, size ->
            val dims = inputShape.toMutableList()
            if (axis in dims.indices) dims[axis] = size
            outputs += TensorSpec(
                name = "split_output_$idx",
                shape = dims,
                dtype = input.dtype
            )
        }
        return outputs
    }
}

private open class BasePlaceholderOperation(
    opName: String,
    parameters: Map<String, Any> = emptyMap(),
    private val opType: String = "onnx"
) : sk.ainet.lang.tensor.ops.BaseOperation(opName, opType, parameters) {
    override fun <T2 : DType, V2> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T2, V2>>): List<sk.ainet.lang.tensor.Tensor<T2, V2>> {
        throw UnsupportedOperationException("Placeholder operation '$name' cannot execute")
    }

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        val first = inputs.firstOrNull()
        val outputSpec = TensorSpec(
            name = "${name}_output",
            shape = first?.shape,
            dtype = first?.dtype ?: "unknown"
        )
        return listOf(outputSpec)
    }

    override fun clone(newParameters: Map<String, Any>): Operation =
        BasePlaceholderOperation(name, newParameters, opType)
}
