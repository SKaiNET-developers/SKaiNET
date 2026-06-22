package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for neural network operations.
 * 
 * This converter handles neural network specific operations including:
 * - Convolutional operations (conv2d) with proper attribute mapping
 * - Pooling operations (maxPool2d, avgPool2d) with correct StableHLO mapping
 * - Batch normalization and layer normalization operations
 * 
 * Supports operations as specified in Requirements 2.4:
 * - conv2d with proper attribute mapping for strides, padding, dilation
 * - Different pooling types (max, average) with correct StableHLO mapping
 * - Batch normalization and layer normalization operations
 */
public class NeuralNetOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        // Convolutional operations
        "conv1d", "conv2d",
        // Pooling operations
        "maxPool2d", "avgPool2d", "averagePool2d",
        // Normalization operations
        "batchNorm", "batchNormalization", "BatchNormalization",
        "layerNorm", "layerNormalization", "LayerNormalization",
        "rmsNorm", "rms_norm", "RMSNorm", "RmsNorm",
        "groupNorm", "groupNormalization", "GroupNormalization", "group_norm",
        // Attention
        "scaledDotProductAttention"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "conv1d" -> convertConv1d(node, operands, context)
            "conv2d" -> convertConv2d(node, operands, context)
            "maxpool2d" -> convertMaxPool2d(node, operands, context)
            "avgpool2d", "averagepool2d" -> convertAvgPool2d(node, operands, context)
            "batchnorm", "batchnormalization" -> convertBatchNorm(node, operands, context)
            "layernorm", "layernormalization" -> convertLayerNorm(node, operands, context)
            "rmsnorm", "rms_norm" -> convertRmsNorm(node, operands, context)
            "groupnorm", "groupnormalization", "group_norm" -> convertGroupNorm(node, operands, context)
            "scaleddotproductattention" -> convertSdpa(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by NeuralNetOperationsConverter"
            )
        }
    }
    
    private fun convertConv1d(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 2) {
            return ConversionResult.Failure(
                "Conv1d operation requires at least 2 operands (input, weight), got ${operands.size}",
                "Unsupported conv1d arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?x?x?xf32>"

        val params = node.operation.parameters
        val stride = params["stride"] as? Int ?: 1
        val padding = params["padding"] as? Int ?: 0
        val dilation = params["dilation"] as? Int ?: 1
        val groups = params["groups"] as? Int ?: 1

        val resultValue = context.nextTempValue()

        val typeMapper = context.getTypeMapper()
        val inputType = node.inputs.getOrNull(0)?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?x?xf32>"
        val weightType = node.inputs.getOrNull(1)?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?x?xf32>"
        val biasType = node.inputs.getOrNull(2)?.let { typeMapper.mapTensorType(it) }

        val convOperation = buildConv1dOperation(
            resultValue = resultValue,
            input = operands[0],
            weight = operands[1],
            bias = if (operands.size > 2) operands[2] else null,
            inputType = inputType,
            weightType = weightType,
            outputType = outputType,
            biasType = biasType,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups
        )

        context.emitOperation(convOperation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(convOperation)
        )
    }

    private fun convertConv2d(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 2) {
            return ConversionResult.Failure(
                "Conv2d operation requires at least 2 operands (input, weight), got ${operands.size}",
                "Unsupported conv2d arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?x?x?xf32>"
        
        // Extract convolution parameters from operation parameters
        val params = node.operation.parameters
        val stride = extractStride(params)
        val padding = extractPadding(params)
        val dilation = extractDilation(params)
        val groups = params["groups"] as? Int ?: 1
        
        val resultValue = context.nextTempValue()

        // Resolve input/weight types for the functional type annotation
        val typeMapper = context.getTypeMapper()
        val inputType = node.inputs.getOrNull(0)?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?x?x?xf32>"
        val weightType = node.inputs.getOrNull(1)?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?x?x?xf32>"
        val biasType = node.inputs.getOrNull(2)?.let { typeMapper.mapTensorType(it) }

        // Build StableHLO convolution operation
        val convOperation = buildConvolutionOperation(
            resultValue = resultValue,
            input = operands[0],
            weight = operands[1],
            bias = if (operands.size > 2) operands[2] else null,
            inputType = inputType,
            weightType = weightType,
            outputType = outputType,
            biasType = biasType,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups
        )
        
        context.emitOperation(convOperation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(convOperation)
        )
    }
    
    private fun convertMaxPool2d(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "MaxPool2d operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported maxPool2d arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?x?x?xf32>"
        
        // Extract pooling parameters
        val params = node.operation.parameters
        val kernelSize = extractKernelSize(params)
        val stride = extractStride(params)
        val padding = extractPadding(params)
        
        val resultValue = context.nextTempValue()
        val inputType = node.inputs.firstOrNull()?.let { context.getTypeMapper().mapTensorType(it) } ?: outputType

        // Build StableHLO reduce_window operation for max pooling
        val operations = buildMaxPoolOperations(
            resultValue = resultValue,
            input = operands[0],
            inputType = inputType,
            outputType = outputType,
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            context = context
        )
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
    
    private fun convertAvgPool2d(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "AvgPool2d operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported avgPool2d arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?x?x?xf32>"
        
        // Extract pooling parameters
        val params = node.operation.parameters
        val kernelSize = extractKernelSize(params)
        val stride = extractStride(params)
        val padding = extractPadding(params)
        
        val resultValue = context.nextTempValue()
        val inputType = node.inputs.firstOrNull()?.let { context.getTypeMapper().mapTensorType(it) } ?: outputType

        // Build StableHLO reduce_window operation for average pooling
        val operations = buildAvgPoolOperations(
            resultValue = resultValue,
            input = operands[0],
            inputType = inputType,
            outputType = outputType,
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            context = context
        )
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
    
    private fun convertBatchNorm(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 3) {
            return ConversionResult.Failure(
                "BatchNorm operation requires at least 3 operands (input, scale, offset), got ${operands.size}",
                "Unsupported batchNorm arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?x?x?xf32>"
        
        // Extract batch norm parameters
        val params = node.operation.parameters
        val epsilon = params["eps"] as? Double ?: 1e-5
        val featureIndex = params["feature_index"] as? Int ?: 1 // Channel dimension
        
        val resultValue = context.nextTempValue()
        
        // Build StableHLO batch_norm_inference operation
        val batchNormOperation = buildBatchNormOperation(
            resultValue = resultValue,
            input = operands[0],
            scale = operands[1],
            offset = operands[2],
            mean = if (operands.size > 3) operands[3] else null,
            variance = if (operands.size > 4) operands[4] else null,
            outputType = outputType,
            epsilon = epsilon,
            featureIndex = featureIndex
        )
        
        context.emitOperation(batchNormOperation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(batchNormOperation)
        )
    }
    
    /**
     * Lower LayerNorm to real StableHLO elementwise ops. Replaces an
     * earlier lowering that emitted `stablehlo.custom_call @layer_norm`
     * as a placeholder (no MLIR tool in the repo understands that
     * custom_call), using the standard decomposition:
     *
     *     out = scale * (x - mean) / sqrt(var + eps) + offset
     *
     * Emission style matches the softmax fix (#467) and the rest of
     * the emitter: reductions go through
     * `stablehlo.custom_call @reduce_mean` / `@reduce_variance` (both
     * already supported by `ReductionOperationsConverter`), the reduced
     * tensors are broadcast back to the input shape via
     * `stablehlo.broadcast_in_dim`, and scale / offset are elementwise
     * multiplied / added only when their operands are actually present.
     * Migrating every reduction to real `stablehlo.reduce` regions is
     * a separate, larger refactor.
     */
    private fun convertLayerNorm(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isEmpty()) {
            return ConversionResult.Failure(
                "LayerNorm operation requires at least 1 operand (input), got ${operands.size}",
                "Unsupported layerNorm arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?x?xf32>"
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) }
            ?: "f32"

        val inputShape = node.inputs.firstOrNull()?.shape ?: outputSpec?.shape ?: emptyList()
        val rank = inputShape.size

        // Normalize the axis parameter against rank. Default to the
        // last dimension, matching every standard LayerNorm in the
        // wild. Callers may supply either an `axis` integer or an
        // `IntArray normalized_shape`; in the latter case we reduce
        // along the leading element (simple single-axis support).
        val rawAxis = node.operation.parameters["axis"] as? Int
            ?: (node.operation.parameters["normalized_shape"] as? IntArray)?.firstOrNull()
            ?: -1
        val axis = when {
            rank == 0 -> 0
            rawAxis < 0 -> rank + rawAxis
            else -> rawAxis
        }.coerceIn(0, (rank - 1).coerceAtLeast(0))

        val reducedShape = if (rank > 0) {
            inputShape.filterIndexed { i, _ -> i != axis }
        } else {
            emptyList()
        }
        val reducedType = if (reducedShape.isEmpty()) {
            "tensor<$elementType>"
        } else {
            "tensor<${reducedShape.joinToString("x")}x$elementType>"
        }
        val broadcastDims = (0 until rank).filter { it != axis }.joinToString(", ")

        val epsilon = (node.operation.parameters["eps"] as? Double)
            ?: (node.operation.parameters["epsilon"] as? Double)
            ?: 1e-5  // LayerNorm family default.

        val xInput = operands[0]
        val scaleOperand: String? = if (operands.size > 1) operands[1] else null
        val offsetOperand: String? = if (operands.size > 2) operands[2] else null

        val meanValue = context.nextTempValue()
        val meanBroadcast = context.nextTempValue()
        val centered = context.nextTempValue()
        val varValue = context.nextTempValue()
        val epsConst = context.nextTempValue()
        val epsBroadcast = context.nextTempValue()
        val varPlusEps = context.nextTempValue()
        val stdValue = context.nextTempValue()
        val stdBroadcast = context.nextTempValue()
        val normalized = context.nextTempValue()

        val operations = mutableListOf<String>()

        // mean(x) along the normalization axis.
        operations += "$meanValue = stablehlo.custom_call @reduce_mean($xInput) " +
            "{dimensions = [$axis], keepdim = false} : $reducedType"

        // Broadcast mean back to input shape.
        operations += "$meanBroadcast = stablehlo.broadcast_in_dim $meanValue, " +
            "dims = [$broadcastDims] : ($reducedType) -> $outputType"

        // Mean-center.
        operations += "$centered = stablehlo.subtract $xInput, $meanBroadcast : $outputType"

        // variance(x) along the normalization axis.
        operations += "$varValue = stablehlo.custom_call @reduce_variance($xInput) " +
            "{dimensions = [$axis], keepdim = false} : $reducedType"

        // Epsilon constant broadcast into the reduced shape.
        operations += "$epsConst = stablehlo.constant dense<$epsilon> : tensor<$elementType>"
        operations += "$epsBroadcast = stablehlo.broadcast_in_dim $epsConst, " +
            "dims = [] : (tensor<$elementType>) -> $reducedType"

        // variance + eps
        operations += "$varPlusEps = stablehlo.add $varValue, $epsBroadcast : $reducedType"

        // std = sqrt(variance + eps)
        operations += "$stdValue = stablehlo.sqrt $varPlusEps : $reducedType"

        // Broadcast std back to the input shape.
        operations += "$stdBroadcast = stablehlo.broadcast_in_dim $stdValue, " +
            "dims = [$broadcastDims] : ($reducedType) -> $outputType"

        // normalized = (x - mean) / std
        operations += "$normalized = stablehlo.divide $centered, $stdBroadcast : $outputType"

        // Apply scale and offset if present. Track the current running
        // SSA value so omitting either one keeps the emitted MLIR
        // faithful to the input graph.
        var current = normalized
        if (scaleOperand != null) {
            val scaled = context.nextTempValue()
            operations += "$scaled = stablehlo.multiply $current, $scaleOperand : $outputType"
            current = scaled
        }
        if (offsetOperand != null) {
            val offsetted = context.nextTempValue()
            operations += "$offsetted = stablehlo.add $current, $offsetOperand : $outputType"
            current = offsetted
        }

        operations.forEach { context.emitOperation(it) }

        return ConversionResult.Success(
            outputValueName = current,
            emittedOperations = operations
        )
    }
    
    /**
     * Lower GroupNorm to real StableHLO elementwise ops, in the same
     * decomposition style as LayerNorm / RMSNorm (no `@group_norm`
     * custom_call stub). GroupNorm splits the `C` channels of an
     * `(N, C, *spatial)` input into `num_groups` groups and normalizes each
     * group over its channels and spatial positions, then applies an
     * optional per-channel affine:
     *
     *     xg   = reshape(x, [N, G, M])              // M = (C/G) * prod(spatial)
     *     out  = (xg - mean(xg)) / sqrt(var(xg) + eps)   // reduce over M
     *     out  = reshape(out, [N, C, *spatial])
     *     out  = out * scale + offset               // scale/offset shape (C,), optional
     *
     * The per-group reduction reuses the single-axis `@reduce_mean` /
     * `@reduce_variance` custom_calls (exactly as LayerNorm does) by collapsing
     * each group's channels + spatial into one trailing axis. Scale and offset
     * broadcast over the channel dimension only.
     */
    private fun convertGroupNorm(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isEmpty()) {
            return ConversionResult.Failure(
                "GroupNorm operation requires at least 1 operand (input), got ${operands.size}",
                "Unsupported groupNorm arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?x?x?x?xf32>"
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) }
            ?: "f32"

        val inputShape = node.inputs.firstOrNull()?.shape ?: outputSpec?.shape ?: emptyList()
        if (inputShape.size < 2) {
            return ConversionResult.Failure(
                "GroupNorm requires an (N, C, ...) input of rank >= 2, got rank ${inputShape.size}",
                "Unsupported groupNorm input rank for node ${node.id}"
            )
        }

        val n = inputShape[0]
        val c = inputShape[1]
        val spatialCount = inputShape.drop(2).fold(1) { acc, d -> acc * d }

        val params = node.operation.parameters
        val numGroups = (params["num_groups"] as? Int)
            ?: (params["groups"] as? Int)
            ?: (params["numGroups"] as? Int)
            ?: 1
        val groups = numGroups.coerceIn(1, if (c > 0) c else 1)
        if (c % groups != 0) {
            return ConversionResult.Failure(
                "GroupNorm channels ($c) must be divisible by num_groups ($groups)",
                "Unsupported groupNorm grouping for node ${node.id}"
            )
        }
        val perGroup = (c / groups) * spatialCount  // M

        val epsilon = (params["eps"] as? Double)
            ?: (params["epsilon"] as? Double)
            ?: 1e-5

        val groupedType = "tensor<${n}x${groups}x${perGroup}x$elementType>"
        val reducedType = "tensor<${n}x${groups}x$elementType>"

        val xInput = operands[0]
        val scaleOperand: String? = if (operands.size > 1) operands[1] else null
        val offsetOperand: String? = if (operands.size > 2) operands[2] else null

        val grouped = context.nextTempValue()
        val meanValue = context.nextTempValue()
        val meanBroadcast = context.nextTempValue()
        val centered = context.nextTempValue()
        val varValue = context.nextTempValue()
        val epsConst = context.nextTempValue()
        val epsBroadcast = context.nextTempValue()
        val varPlusEps = context.nextTempValue()
        val stdValue = context.nextTempValue()
        val stdBroadcast = context.nextTempValue()
        val normalized = context.nextTempValue()
        val reshapedBack = context.nextTempValue()

        val operations = mutableListOf<String>()

        // Reshape (N, C, *spatial) -> (N, G, M): collapse each group's channels +
        // spatial into one trailing axis so a single-axis reduction is per-group.
        operations += "$grouped = stablehlo.reshape $xInput : ($outputType) -> $groupedType"

        // mean(xg) over the trailing axis, broadcast back, mean-center.
        operations += "$meanValue = stablehlo.custom_call @reduce_mean($grouped) " +
            "{dimensions = [2], keepdim = false} : $reducedType"
        operations += "$meanBroadcast = stablehlo.broadcast_in_dim $meanValue, " +
            "dims = [0, 1] : ($reducedType) -> $groupedType"
        operations += "$centered = stablehlo.subtract $grouped, $meanBroadcast : $groupedType"

        // var(xg) over the trailing axis; std = sqrt(var + eps).
        operations += "$varValue = stablehlo.custom_call @reduce_variance($grouped) " +
            "{dimensions = [2], keepdim = false} : $reducedType"
        operations += "$epsConst = stablehlo.constant dense<$epsilon> : tensor<$elementType>"
        operations += "$epsBroadcast = stablehlo.broadcast_in_dim $epsConst, " +
            "dims = [] : (tensor<$elementType>) -> $reducedType"
        operations += "$varPlusEps = stablehlo.add $varValue, $epsBroadcast : $reducedType"
        operations += "$stdValue = stablehlo.sqrt $varPlusEps : $reducedType"
        operations += "$stdBroadcast = stablehlo.broadcast_in_dim $stdValue, " +
            "dims = [0, 1] : ($reducedType) -> $groupedType"
        operations += "$normalized = stablehlo.divide $centered, $stdBroadcast : $groupedType"

        // Reshape back to (N, C, *spatial).
        operations += "$reshapedBack = stablehlo.reshape $normalized : ($groupedType) -> $outputType"

        // Optional per-channel affine: scale/offset have shape (C,), broadcast over
        // the channel dimension (index 1).
        var current = reshapedBack
        if (scaleOperand != null) {
            val scaleBroadcast = context.nextTempValue()
            val scaled = context.nextTempValue()
            operations += "$scaleBroadcast = stablehlo.broadcast_in_dim $scaleOperand, " +
                "dims = [1] : (tensor<${c}x$elementType>) -> $outputType"
            operations += "$scaled = stablehlo.multiply $current, $scaleBroadcast : $outputType"
            current = scaled
        }
        if (offsetOperand != null) {
            val offsetBroadcast = context.nextTempValue()
            val offsetted = context.nextTempValue()
            operations += "$offsetBroadcast = stablehlo.broadcast_in_dim $offsetOperand, " +
                "dims = [1] : (tensor<${c}x$elementType>) -> $outputType"
            operations += "$offsetted = stablehlo.add $current, $offsetBroadcast : $outputType"
            current = offsetted
        }

        operations.forEach { context.emitOperation(it) }

        return ConversionResult.Success(
            outputValueName = current,
            emittedOperations = operations
        )
    }

    /**
     * Lower RMSNorm to real StableHLO elementwise ops. This is the
     * normalization every Llama / Mistral / Qwen / Gemma family
     * transformer uses — it drops the mean-centering and the additive
     * offset of LayerNorm, leaving:
     *
     *     rms  = sqrt(mean(x^2, axis) + eps)
     *     out  = scale * x / rms      (scale operand is optional)
     *
     * The reductions are emitted as `stablehlo.custom_call @reduce_mean`
     * to match the style already used by `ReductionOperationsConverter`
     * and by the softmax lowering in `ActivationOperationsConverter`.
     * Migrating every reduction to proper `stablehlo.reduce` regions is
     * a separate, larger refactor.
     */
    private fun convertRmsNorm(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isEmpty()) {
            return ConversionResult.Failure(
                "RMSNorm operation requires at least 1 operand (input), got ${operands.size}",
                "Unsupported rmsNorm arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) }
            ?: "f32"

        val inputShape = node.inputs.firstOrNull()?.shape ?: outputSpec?.shape ?: emptyList()
        val rank = inputShape.size

        // Normalize the axis parameter against rank. Default to the
        // last dimension, consistent with softmax and every LLM RMSNorm
        // implementation in the wild.
        val rawAxis = node.operation.parameters["axis"] as? Int
            ?: (node.operation.parameters["normalized_shape"] as? IntArray)?.firstOrNull()
            ?: -1
        val axis = when {
            rank == 0 -> 0
            rawAxis < 0 -> rank + rawAxis
            else -> rawAxis
        }.coerceIn(0, (rank - 1).coerceAtLeast(0))

        // Reduced tensor shape: input with `axis` removed. Matches the
        // same reduced-type convention `convertSoftmax` uses.
        val reducedShape = if (rank > 0) {
            inputShape.filterIndexed { i, _ -> i != axis }
        } else {
            emptyList()
        }
        val reducedType = if (reducedShape.isEmpty()) {
            "tensor<$elementType>"
        } else {
            "tensor<${reducedShape.joinToString("x")}x$elementType>"
        }

        // Dimensions kept for broadcast_in_dim: every input dim except
        // `axis`, mapped to its position in the reduced tensor.
        val broadcastDims = (0 until rank).filter { it != axis }.joinToString(", ")

        val eps = (node.operation.parameters["eps"] as? Double)
            ?: (node.operation.parameters["epsilon"] as? Double)
            ?: 1e-6  // Llama family default; LayerNorm typically uses 1e-5

        val xInput = operands[0]
        val scaleOperand: String? = if (operands.size >= 2) operands[1] else null

        val xSquared = context.nextTempValue()
        val meanSquared = context.nextTempValue()
        val epsConst = context.nextTempValue()
        val epsBroadcast = context.nextTempValue()
        val meanPlusEps = context.nextTempValue()
        val rms = context.nextTempValue()
        val rmsBroadcast = context.nextTempValue()
        val normalized = context.nextTempValue()
        val resultValue = context.nextTempValue()

        val operations = mutableListOf<String>()

        // x^2
        operations += "$xSquared = stablehlo.multiply $xInput, $xInput : $outputType"

        // reduce_mean(x^2, axis)
        operations += "$meanSquared = stablehlo.custom_call @reduce_mean($xSquared) " +
            "{dimensions = [$axis], keepdim = false} : $reducedType"

        // eps constant broadcast into the reduced shape
        operations += "$epsConst = stablehlo.constant dense<$eps> : tensor<$elementType>"
        operations += "$epsBroadcast = stablehlo.broadcast_in_dim $epsConst, " +
            "dims = [] : (tensor<$elementType>) -> $reducedType"

        // mean + eps
        operations += "$meanPlusEps = stablehlo.add $meanSquared, $epsBroadcast : $reducedType"

        // rms = sqrt(mean + eps)
        operations += "$rms = stablehlo.sqrt $meanPlusEps : $reducedType"

        // Broadcast rms back to the input shape for the elementwise divide.
        operations += "$rmsBroadcast = stablehlo.broadcast_in_dim $rms, " +
            "dims = [$broadcastDims] : ($reducedType) -> $outputType"

        // x / rms
        operations += "$normalized = stablehlo.divide $xInput, $rmsBroadcast : $outputType"

        // Final scale multiply is optional — when the caller did not
        // pass a scale operand we return the normalized value directly.
        val finalValue: String
        if (scaleOperand != null) {
            operations += "$resultValue = stablehlo.multiply $normalized, $scaleOperand : $outputType"
            finalValue = resultValue
        } else {
            finalValue = normalized
        }

        operations.forEach { context.emitOperation(it) }

        return ConversionResult.Success(
            outputValueName = finalValue,
            emittedOperations = operations
        )
    }

    // Helper functions for parameter extraction
    
    private fun extractStride(params: Map<String, Any>): Pair<Int, Int> {
        return when (val stride = params["stride"]) {
            is Pair<*, *> -> (stride.first as? Int ?: 1) to (stride.second as? Int ?: 1)
            is Int -> stride to stride
            is List<*> -> {
                val list = stride.filterIsInstance<Int>()
                if (list.size >= 2) list[0] to list[1] else 1 to 1
            }
            else -> 1 to 1
        }
    }
    
    private fun extractPadding(params: Map<String, Any>): Pair<Int, Int> {
        return when (val padding = params["padding"]) {
            is Pair<*, *> -> (padding.first as? Int ?: 0) to (padding.second as? Int ?: 0)
            is Int -> padding to padding
            is List<*> -> {
                val list = padding.filterIsInstance<Int>()
                if (list.size >= 2) list[0] to list[1] else 0 to 0
            }
            else -> 0 to 0
        }
    }
    
    private fun extractDilation(params: Map<String, Any>): Pair<Int, Int> {
        return when (val dilation = params["dilation"]) {
            is Pair<*, *> -> (dilation.first as? Int ?: 1) to (dilation.second as? Int ?: 1)
            is Int -> dilation to dilation
            is List<*> -> {
                val list = dilation.filterIsInstance<Int>()
                if (list.size >= 2) list[0] to list[1] else 1 to 1
            }
            else -> 1 to 1
        }
    }
    
    private fun extractKernelSize(params: Map<String, Any>): Pair<Int, Int> {
        return when (val kernelSize = params["kernelSize"] ?: params["kernel_size"]) {
            is Pair<*, *> -> (kernelSize.first as? Int ?: 2) to (kernelSize.second as? Int ?: 2)
            is Int -> kernelSize to kernelSize
            is List<*> -> {
                val list = kernelSize.filterIsInstance<Int>()
                if (list.size >= 2) list[0] to list[1] else 2 to 2
            }
            else -> 2 to 2
        }
    }
    
    // Helper functions for building StableHLO operations
    
    private fun buildConv1dOperation(
        resultValue: String,
        input: String,
        weight: String,
        bias: String?,
        inputType: String,
        weightType: String,
        outputType: String,
        biasType: String?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): String {
        // StableHLO convolution with 1D spatial dimensions:
        //   dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0]
        //   (batch=implicit, feature, one spatial dim)
        val convCore = { rv: String ->
            "$rv = stablehlo.convolution($input, $weight) " +
                    "dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0], " +
                    "window = {stride = [$stride], " +
                    "pad = [[$padding, $padding]], " +
                    "rhs_dilate = [$dilation]} " +
                    "{batch_group_count = 1 : i64, feature_group_count = $groups : i64} " +
                    ": ($inputType, $weightType) -> $outputType"
        }

        return if (bias != null) {
            val convResult = "${resultValue}_conv"
            val biasBcast = "${resultValue}_bias_b"
            val convOp = convCore(convResult)
            // bias is [Cout]; conv output is [N, Cout, L]; broadcast along feature dim (index 1)
            val bcastOp = "$biasBcast = stablehlo.broadcast_in_dim $bias, dims = [1] : " +
                    "(${biasType ?: "tensor<?xf32>"}) -> $outputType"
            "$convOp\n    $bcastOp\n    $resultValue = stablehlo.add $convResult, $biasBcast : $outputType"
        } else {
            convCore(resultValue)
        }
    }

    private fun buildConvolutionOperation(
        resultValue: String,
        input: String,
        weight: String,
        bias: String?,
        inputType: String,
        weightType: String,
        outputType: String,
        biasType: String?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): String {
        // StableHLO convolution custom assembly format:
        //   %r = stablehlo.convolution(%lhs, %rhs)
        //     dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
        //     window = {stride = [...], pad = [...], rhs_dilate = [...]}
        //     {batch_group_count = 1 : i64, feature_group_count = N : i64}
        //     : (lhs_type, rhs_type) -> result_type
        val convCore = { rv: String ->
            "$rv = stablehlo.convolution($input, $weight) " +
                    "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], " +
                    "window = {stride = [${stride.first}, ${stride.second}], " +
                    "pad = [[${padding.first}, ${padding.first}], [${padding.second}, ${padding.second}]], " +
                    "rhs_dilate = [${dilation.first}, ${dilation.second}]} " +
                    "{batch_group_count = 1 : i64, feature_group_count = $groups : i64} " +
                    ": ($inputType, $weightType) -> $outputType"
        }

        return if (bias != null) {
            val convResult = "${resultValue}_conv"
            val biasBcast = "${resultValue}_bias_b"
            val convOp = convCore(convResult)
            // bias is [Cout]; conv output is [N, Cout, H, W]; broadcast along feature dim (index 1)
            val bcastOp = "$biasBcast = stablehlo.broadcast_in_dim $bias, dims = [1] : " +
                    "(${biasType ?: "tensor<?xf32>"}) -> $outputType"
            "$convOp\n    $bcastOp\n    $resultValue = stablehlo.add $convResult, $biasBcast : $outputType"
        } else {
            convCore(resultValue)
        }
    }

    /** The MLIR element type ("f32"/"f16"/…) parsed from a `tensor<…xT>` string. */
    private fun elementTypeOf(tensorType: String): String =
        tensorType.substringAfterLast('x').substringBefore('>').ifBlank { "f32" }

    /**
     * Emit a `reduce_window` in IREE's parseable **generic region** form. The pretty
     * `… applies <op> over window dimensions = …` form is rejected by IREE's StableHLO
     * parser ("has no custom assembly form"), and its 2-element window only covered H/W;
     * the generic form carries full NCHW-rank (`[1, 1, kH, kW]`) window attributes. (#675)
     */
    private fun reduceWindowGeneric(
        resultValue: String,
        input: String,
        inputType: String,
        initValue: String,
        reduceOp: String,
        elem: String,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        outputType: String,
    ): String {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (pH, pW) = padding
        // Single line: MLIR treats newlines as whitespace, and the line-based MLIR
        // validator only handles one op per line. The region body ops are separated
        // by spaces, which the MLIR parser accepts.
        // Region-local SSA names are derived from the (unique) result value so two
        // pooling ops in one function don't collide in the flat validator (they are
        // region-scoped in MLIR, but the validator tracks names globally).
        val t = resultValue.removePrefix("%")
        return "$resultValue = \"stablehlo.reduce_window\"($input, $initValue) ({ " +
            "^bb0(%lhs_$t: tensor<$elem>, %rhs_$t: tensor<$elem>): " +
            "%out_$t = $reduceOp %lhs_$t, %rhs_$t : tensor<$elem> " +
            "stablehlo.return %out_$t : tensor<$elem> " +
            "}) {window_dimensions = array<i64: 1, 1, $kH, $kW>, " +
            "window_strides = array<i64: 1, 1, $sH, $sW>, " +
            "base_dilations = array<i64: 1, 1, 1, 1>, " +
            "window_dilations = array<i64: 1, 1, 1, 1>, " +
            "padding = dense<[[0, 0], [0, 0], [$pH, $pH], [$pW, $pW]]> : tensor<4x2xi64>} : " +
            "($inputType, tensor<$elem>) -> $outputType"
    }

    private fun buildMaxPoolOperations(
        resultValue: String,
        input: String,
        inputType: String,
        outputType: String,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        context: ConversionContext
    ): List<String> {
        val elem = elementTypeOf(outputType)
        val initValue = context.nextTempValue()
        val initConstant = "$initValue = stablehlo.constant dense<-3.4028235e+38> : tensor<$elem>"
        val poolOp = reduceWindowGeneric(
            resultValue, input, inputType, initValue, "stablehlo.maximum",
            elem, kernelSize, stride, padding, outputType,
        )
        context.emitOperation(initConstant)
        context.emitOperation(poolOp)
        return listOf(initConstant, poolOp)
    }

    private fun buildAvgPoolOperations(
        resultValue: String,
        input: String,
        inputType: String,
        outputType: String,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        context: ConversionContext
    ): List<String> {
        // Average pooling requires sum + division by kernel size.
        val elem = elementTypeOf(outputType)
        val kernelArea = kernelSize.first * kernelSize.second
        val initZero = context.nextTempValue()
        val kernelAreaConst = context.nextTempValue()
        val sumResult = context.nextTempValue()

        val initConstant = "$initZero = stablehlo.constant dense<0.0> : tensor<$elem>"
        // Splat over the output type so the divide is element-type consistent (a scalar
        // tensor<f32> divisor was a latent type mismatch).
        val areaConstant = "$kernelAreaConst = stablehlo.constant dense<$kernelArea.0> : $outputType"
        val sumOp = reduceWindowGeneric(
            sumResult, input, inputType, initZero, "stablehlo.add",
            elem, kernelSize, stride, padding, outputType,
        )
        val divideOp = "$resultValue = stablehlo.divide $sumResult, $kernelAreaConst : $outputType"

        context.emitOperation(initConstant)
        context.emitOperation(areaConstant)
        context.emitOperation(sumOp)
        context.emitOperation(divideOp)
        return listOf(initConstant, areaConstant, sumOp, divideOp)
    }
    
    private fun buildBatchNormOperation(
        resultValue: String,
        input: String,
        scale: String,
        offset: String,
        mean: String?,
        variance: String?,
        outputType: String,
        epsilon: Double,
        featureIndex: Int
    ): String {
        return if (mean != null && variance != null) {
            // Use batch_norm_inference when mean and variance are provided
            "$resultValue = stablehlo.batch_norm_inference $input, $scale, $offset, $mean, $variance, " +
                    "epsilon = $epsilon, feature_index = $featureIndex : $outputType"
        } else {
            // Use batch_norm_training when mean and variance need to be computed
            "$resultValue = stablehlo.batch_norm_training $input, $scale, $offset, " +
                    "epsilon = $epsilon, feature_index = $featureIndex : $outputType"
        }
    }

    /**
     * Convert scaledDotProductAttention to StableHLO.
     * Decomposes into: Q @ K.T (batched) → scale → optional mask → softmax → @ V (batched)
     *
     * Input shapes: Q[B,H,S,D], K[B,H,T,D], V[B,H,T,D], optional mask[B,H,S,T] or broadcastable
     * Output: [B,H,S,D]
     */
    private fun convertSdpa(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 3) {
            return ConversionResult.Failure("SDPA requires at least 3 operands (Q, K, V), got ${operands.size}")
        }

        val query = operands[0]    // [B, H, S, D]
        val key = operands[1]      // [B, H, T, D]
        val value = operands[2]    // [B, H, T, D]
        val mask = if (operands.size >= 4) operands[3] else null

        val querySpec = node.inputs.getOrNull(0)
        val keySpec = node.inputs.getOrNull(1)
        val valueSpec = node.inputs.getOrNull(2)

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        // Infer shapes for intermediate types
        val qShape = querySpec?.shape ?: return ConversionResult.Failure("Unknown Q shape")
        val kShape = keySpec?.shape ?: return ConversionResult.Failure("Unknown K shape")
        val vShape = valueSpec?.shape ?: return ConversionResult.Failure("Unknown V shape")

        val rank = qShape.size
        if (rank != 4) {
            return ConversionResult.Failure("SDPA expects 4D tensors [B,H,S,D], got rank $rank")
        }

        val batch = qShape[0]
        val heads = qShape[1]
        val seqQ = qShape[2]
        val headDim = qShape[3]
        val seqK = kShape[2]

        val queryType = context.getValueType(query) ?: "tensor<${qShape.joinToString("x")}xf32>"
        val keyType = context.getValueType(key) ?: "tensor<${kShape.joinToString("x")}xf32>"
        val valueType = context.getValueType(value) ?: "tensor<${vShape.joinToString("x")}xf32>"

        // scores = Q @ K.T: [B,H,S,D] @ [B,H,T,D] → [B,H,S,T]
        // dot_general with batching_dims=[0,1], contracting_dims=[3]x[3]
        val scoresType = "tensor<${batch}x${heads}x${seqQ}x${seqK}xf32>"
        val scoresVal = context.nextTempValue()
        context.emitOperation(
            "$scoresVal = stablehlo.dot_general $query, $key, " +
            "batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] " +
            ": ($queryType, $keyType) -> $scoresType"
        )
        context.setValueType(scoresVal, scoresType)

        // Scale
        val scale = node.operation.parameters["scale"] as? Float
            ?: (1.0f / kotlin.math.sqrt(headDim.toFloat()))
        val scaledVal = context.nextTempValue()
        val scaleConst = context.nextTempValue()
        context.emitOperation("$scaleConst = stablehlo.constant dense<$scale> : tensor<f32>")
        context.emitOperation(
            "$scaledVal = stablehlo.broadcast_in_dim $scaleConst, dims = [] " +
            ": (tensor<f32>) -> $scoresType"
        )
        val scaledScores = context.nextTempValue()
        context.emitOperation(
            "$scaledScores = stablehlo.multiply $scoresVal, $scaledVal : $scoresType"
        )
        context.setValueType(scaledScores, scoresType)

        // Optional mask
        var presoft = scaledScores
        if (mask != null) {
            val maskedVal = context.nextTempValue()
            val maskType = context.getValueType(mask) ?: scoresType
            context.emitOperation(
                "$maskedVal = stablehlo.add $presoft, $mask : $scoresType"
            )
            context.setValueType(maskedVal, scoresType)
            presoft = maskedVal
        }

        // Softmax over last dim (seqK)
        // Decompose: exp(x - max(x)) / sum(exp(x - max(x)))
        val maxVal = context.nextTempValue()
        val maxInitVal = context.nextTempValue()
        context.emitOperation("$maxInitVal = stablehlo.constant dense<0xFF800000> : tensor<f32>") // -inf
        context.emitOperation(
            "$maxVal = stablehlo.reduce($presoft init: $maxInitVal) applies stablehlo.maximum " +
            "across dimensions = [${rank - 1}] : ($scoresType, tensor<f32>) -> " +
            "tensor<${batch}x${heads}x${seqQ}xf32>"
        )

        val maxBcast = context.nextTempValue()
        val reducedType = "tensor<${batch}x${heads}x${seqQ}xf32>"
        context.emitOperation(
            "$maxBcast = stablehlo.broadcast_in_dim $maxVal, dims = [0, 1, 2] " +
            ": ($reducedType) -> $scoresType"
        )

        val shifted = context.nextTempValue()
        context.emitOperation("$shifted = stablehlo.subtract $presoft, $maxBcast : $scoresType")

        val expVal = context.nextTempValue()
        context.emitOperation("$expVal = stablehlo.exponential $shifted : $scoresType")

        val sumInit = context.nextTempValue()
        context.emitOperation("$sumInit = stablehlo.constant dense<0.0> : tensor<f32>")
        val sumVal = context.nextTempValue()
        context.emitOperation(
            "$sumVal = stablehlo.reduce($expVal init: $sumInit) applies stablehlo.add " +
            "across dimensions = [${rank - 1}] : ($scoresType, tensor<f32>) -> $reducedType"
        )

        val sumBcast = context.nextTempValue()
        context.emitOperation(
            "$sumBcast = stablehlo.broadcast_in_dim $sumVal, dims = [0, 1, 2] " +
            ": ($reducedType) -> $scoresType"
        )

        val weightsVal = context.nextTempValue()
        context.emitOperation("$weightsVal = stablehlo.divide $expVal, $sumBcast : $scoresType")
        context.setValueType(weightsVal, scoresType)

        // output = weights @ V: [B,H,S,T] @ [B,H,T,D] → [B,H,S,D]
        val resultValue = context.nextTempValue()
        context.emitOperation(
            "$resultValue = stablehlo.dot_general $weightsVal, $value, " +
            "batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] " +
            ": ($scoresType, $valueType) -> $outputType"
        )

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = emptyList()
        )
    }

}