package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode
import kotlin.math.floor

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
        "scaledDotProductAttention",
        // Upsampling / interpolation (Nearest + Bilinear)
        "upsample2d", "Upsample2d", "upsample_2d"
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
            "upsample2d", "upsample_2d" -> convertUpsample2d(node, operands, context)
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
    
    /**
     * Lower BatchNorm to real StableHLO elementwise ops, in the same decomposition style as
     * LayerNorm / GroupNorm — instead of `stablehlo.batch_norm_inference` /
     * `batch_norm_training` (the training form returns a 3-tuple, which the string emitter
     * cannot represent as a single SSA value). Per-channel affine over the `feature_index`
     * (channel) axis of an `(N, C, *spatial)` input:
     *
     *     out = (x - mean) / sqrt(var + eps) * scale + offset
     *
     * `scale` / `offset` (and `mean` / `var` when provided) are shape `(C,)` and broadcast
     * over the channel axis. With 5 operands (input, scale, offset, mean, variance) this is
     * the **inference / eval** form (running statistics). With 3 (input, scale, offset) the
     * batch statistics are computed via real `stablehlo.reduce` over the non-channel axes
     * (population variance, ddof=0) — the **training** form's normalized output.
     */
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
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) }
            ?: "f32"

        val inputShape = node.inputs.firstOrNull()?.shape ?: outputSpec?.shape ?: emptyList()
        val rank = inputShape.size

        val params = node.operation.parameters
        val epsilon = (params["eps"] as? Double) ?: (params["epsilon"] as? Double) ?: 1e-5
        val rawFeature = params["feature_index"] as? Int ?: 1 // channel dimension
        val featureIndex = (if (rawFeature < 0) rank + rawFeature else rawFeature)
            .coerceIn(0, (rank - 1).coerceAtLeast(0))
        val channels = if (rank > 0) inputShape[featureIndex] else 1
        val channelType = "tensor<${channels}x$elementType>"

        val xInput = operands[0]
        val scaleOperand = operands[1]
        val offsetOperand = operands[2]
        val meanOperand = if (operands.size > 3) operands[3] else null
        val varOperand = if (operands.size > 4) operands[4] else null

        val operations = mutableListOf<String>()

        // Per-channel mean & variance: provided directly (inference) or computed via real
        // `stablehlo.reduce` over the non-channel axes (training). Both are shape (C,).
        val meanCh: String
        val varCh: String
        if (meanOperand != null && varOperand != null) {
            meanCh = meanOperand
            varCh = varOperand
        } else {
            val reduceDims = (0 until rank).filter { it != featureIndex }
            val count = reduceDims.fold(1) { acc, d -> acc * inputShape[d] }
            val dimsList = reduceDims.joinToString(", ")
            val zeroInit = context.nextTempValue()
            val countConst = context.nextTempValue()
            val sumX = context.nextTempValue()
            val computedMean = context.nextTempValue()
            val squared = context.nextTempValue()
            val sumSq = context.nextTempValue()
            val meanSq = context.nextTempValue()
            val meanSquared = context.nextTempValue()
            val computedVar = context.nextTempValue()
            operations += "$zeroInit = stablehlo.constant dense<0.0> : tensor<$elementType>"
            operations += "$countConst = stablehlo.constant dense<${count}.0> : $channelType"
            operations += "$sumX = stablehlo.reduce($xInput init: $zeroInit) " +
                "applies stablehlo.add across dimensions = [$dimsList] : ($outputType, tensor<$elementType>) -> $channelType"
            operations += "$computedMean = stablehlo.divide $sumX, $countConst : $channelType"
            operations += "$squared = stablehlo.multiply $xInput, $xInput : $outputType"
            operations += "$sumSq = stablehlo.reduce($squared init: $zeroInit) " +
                "applies stablehlo.add across dimensions = [$dimsList] : ($outputType, tensor<$elementType>) -> $channelType"
            operations += "$meanSq = stablehlo.divide $sumSq, $countConst : $channelType"
            operations += "$meanSquared = stablehlo.multiply $computedMean, $computedMean : $channelType"
            operations += "$computedVar = stablehlo.subtract $meanSq, $meanSquared : $channelType"
            meanCh = computedMean
            varCh = computedVar
        }

        // std = sqrt(var + eps) per channel.
        val epsConst = context.nextTempValue()
        val varPlusEps = context.nextTempValue()
        val stdCh = context.nextTempValue()
        operations += "$epsConst = stablehlo.constant dense<$epsilon> : $channelType"
        operations += "$varPlusEps = stablehlo.add $varCh, $epsConst : $channelType"
        operations += "$stdCh = stablehlo.sqrt $varPlusEps : $channelType"

        // Broadcast the (C,) tensors over the channel axis and apply the affine.
        val meanB = context.nextTempValue()
        val centered = context.nextTempValue()
        val stdB = context.nextTempValue()
        val normalized = context.nextTempValue()
        val scaleB = context.nextTempValue()
        val scaled = context.nextTempValue()
        val offsetB = context.nextTempValue()
        val resultValue = context.nextTempValue()
        operations += "$meanB = stablehlo.broadcast_in_dim $meanCh, " +
            "dims = [$featureIndex] : ($channelType) -> $outputType"
        operations += "$centered = stablehlo.subtract $xInput, $meanB : $outputType"
        operations += "$stdB = stablehlo.broadcast_in_dim $stdCh, " +
            "dims = [$featureIndex] : ($channelType) -> $outputType"
        operations += "$normalized = stablehlo.divide $centered, $stdB : $outputType"
        operations += "$scaleB = stablehlo.broadcast_in_dim $scaleOperand, " +
            "dims = [$featureIndex] : ($channelType) -> $outputType"
        operations += "$scaled = stablehlo.multiply $normalized, $scaleB : $outputType"
        operations += "$offsetB = stablehlo.broadcast_in_dim $offsetOperand, " +
            "dims = [$featureIndex] : ($channelType) -> $outputType"
        operations += "$resultValue = stablehlo.add $scaled, $offsetB : $outputType"

        operations.forEach { context.emitOperation(it) }

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
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
     * Reductions use real `stablehlo.reduce` (sum / count) — not the
     * `@reduce_mean` / `@reduce_variance` custom_call stubs — so the
     * module compiles on stock IREE (matching `convertGroupNorm`).
     * Variance is population (ddof=0) via `E[x²] - E[x]²`. The reduced
     * mean / std are broadcast back to the input shape via
     * `stablehlo.broadcast_in_dim`; scale / offset (shape `[axisSize]`)
     * are broadcast over the normalization axis and applied elementwise
     * only when their operands are actually present.
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

        // Number of elements reduced over the normalization axis (mean/var divisor).
        val axisSize = if (rank > 0) inputShape[axis] else 1
        val scaleType = "tensor<${axisSize}x$elementType>"

        val zeroInit = context.nextTempValue()
        val countConst = context.nextTempValue()
        val sumX = context.nextTempValue()
        val meanValue = context.nextTempValue()
        val meanBroadcast = context.nextTempValue()
        val centered = context.nextTempValue()
        val squared = context.nextTempValue()
        val sumSq = context.nextTempValue()
        val meanSq = context.nextTempValue()
        val meanSquared = context.nextTempValue()
        val varValue = context.nextTempValue()
        val epsConst = context.nextTempValue()
        val epsBroadcast = context.nextTempValue()
        val varPlusEps = context.nextTempValue()
        val stdValue = context.nextTempValue()
        val stdBroadcast = context.nextTempValue()
        val normalized = context.nextTempValue()

        val operations = mutableListOf<String>()

        // Compute the numerically-sensitive normalization (mean / variance / std /
        // divide) in f32 regardless of the model dtype. This is standard LayerNorm
        // practice — PyTorch and JAX upcast fp16/bf16 LayerNorm to f32 internally —
        // because a bf16 variance (a sum of `axisSize` bf16 squares) loses enough
        // precision that `sqrt(var + eps)` can overflow/NaN, and some accelerator
        // backends miscompile the decomposed bf16 reduce/normalize outright. Only the
        // scale/offset affine stays in the model dtype. No-op when the model is f32.
        val computeElement = "f32"
        val isMixed = elementType != computeElement
        val computeType = if (rank > 0) {
            "tensor<${inputShape.joinToString("x")}x$computeElement>"
        } else {
            "tensor<$computeElement>"
        }
        val computeReducedType = if (reducedShape.isEmpty()) {
            "tensor<$computeElement>"
        } else {
            "tensor<${reducedShape.joinToString("x")}x$computeElement>"
        }
        val xF32 = if (isMixed) context.nextTempValue() else xInput
        if (isMixed) {
            operations += "$xF32 = stablehlo.convert $xInput : ($outputType) -> $computeType"
        }

        // mean(x) along the normalization axis, via real `stablehlo.reduce` (sum / count)
        // so the module compiles on stock IREE (no @reduce_* custom_call stubs).
        operations += "$zeroInit = stablehlo.constant dense<0.0> : tensor<$computeElement>"
        operations += "$countConst = stablehlo.constant dense<${axisSize}.0> : $computeReducedType"
        operations += "$sumX = stablehlo.reduce($xF32 init: $zeroInit) " +
            "applies stablehlo.add across dimensions = [$axis] : ($computeType, tensor<$computeElement>) -> $computeReducedType"
        operations += "$meanValue = stablehlo.divide $sumX, $countConst : $computeReducedType"

        // Broadcast mean back to input shape, then mean-center.
        operations += "$meanBroadcast = stablehlo.broadcast_in_dim $meanValue, " +
            "dims = [$broadcastDims] : ($computeReducedType) -> $computeType"
        operations += "$centered = stablehlo.subtract $xF32, $meanBroadcast : $computeType"

        // var(x) = E[x²] - E[x]² (population, ddof=0), again via real reductions.
        operations += "$squared = stablehlo.multiply $xF32, $xF32 : $computeType"
        operations += "$sumSq = stablehlo.reduce($squared init: $zeroInit) " +
            "applies stablehlo.add across dimensions = [$axis] : ($computeType, tensor<$computeElement>) -> $computeReducedType"
        operations += "$meanSq = stablehlo.divide $sumSq, $countConst : $computeReducedType"
        operations += "$meanSquared = stablehlo.multiply $meanValue, $meanValue : $computeReducedType"
        operations += "$varValue = stablehlo.subtract $meanSq, $meanSquared : $computeReducedType"

        // Epsilon constant broadcast into the reduced shape.
        operations += "$epsConst = stablehlo.constant dense<$epsilon> : tensor<$computeElement>"
        operations += "$epsBroadcast = stablehlo.broadcast_in_dim $epsConst, " +
            "dims = [] : (tensor<$computeElement>) -> $computeReducedType"

        // variance + eps
        operations += "$varPlusEps = stablehlo.add $varValue, $epsBroadcast : $computeReducedType"

        // std = sqrt(variance + eps)
        operations += "$stdValue = stablehlo.sqrt $varPlusEps : $computeReducedType"

        // Broadcast std back to the input shape.
        operations += "$stdBroadcast = stablehlo.broadcast_in_dim $stdValue, " +
            "dims = [$broadcastDims] : ($computeReducedType) -> $computeType"

        // normalized = (x - mean) / std  (in f32), then cast back to the model dtype
        // before the scale/offset affine.
        val normalizedF32 = if (isMixed) context.nextTempValue() else normalized
        operations += "$normalizedF32 = stablehlo.divide $centered, $stdBroadcast : $computeType"
        if (isMixed) {
            operations += "$normalized = stablehlo.convert $normalizedF32 : ($computeType) -> $outputType"
        }

        // Apply scale and offset if present. Each has shape [axisSize] and is broadcast over
        // the normalization axis before the elementwise op. Track the running SSA value so
        // omitting either one keeps the emitted MLIR faithful to the input graph.
        var current = normalized
        if (scaleOperand != null) {
            val scaleBroadcast = context.nextTempValue()
            val scaled = context.nextTempValue()
            operations += "$scaleBroadcast = stablehlo.broadcast_in_dim $scaleOperand, " +
                "dims = [$axis] : ($scaleType) -> $outputType"
            operations += "$scaled = stablehlo.multiply $current, $scaleBroadcast : $outputType"
            current = scaled
        }
        if (offsetOperand != null) {
            val offsetBroadcast = context.nextTempValue()
            val offsetted = context.nextTempValue()
            operations += "$offsetBroadcast = stablehlo.broadcast_in_dim $offsetOperand, " +
                "dims = [$axis] : ($scaleType) -> $outputType"
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
        val zeroInit = context.nextTempValue()
        val countConst = context.nextTempValue()
        val sumX = context.nextTempValue()
        val meanValue = context.nextTempValue()
        val meanBroadcast = context.nextTempValue()
        val centered = context.nextTempValue()
        val squared = context.nextTempValue()
        val sumSq = context.nextTempValue()
        val meanSq = context.nextTempValue()
        val meanSquared = context.nextTempValue()
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

        // Reductions use real `stablehlo.reduce` (not @reduce_* custom_calls) so the module
        // compiles on stock IREE. mean(xg) = sum(xg) / M over the trailing axis; broadcast
        // back and mean-center.
        operations += "$zeroInit = stablehlo.constant dense<0.0> : tensor<$elementType>"
        operations += "$countConst = stablehlo.constant dense<${perGroup}.0> : $reducedType"
        operations += "$sumX = stablehlo.reduce($grouped init: $zeroInit) " +
            "applies stablehlo.add across dimensions = [2] : ($groupedType, tensor<$elementType>) -> $reducedType"
        operations += "$meanValue = stablehlo.divide $sumX, $countConst : $reducedType"
        operations += "$meanBroadcast = stablehlo.broadcast_in_dim $meanValue, " +
            "dims = [0, 1] : ($reducedType) -> $groupedType"
        operations += "$centered = stablehlo.subtract $grouped, $meanBroadcast : $groupedType"

        // var(xg) = E[xg^2] - E[xg]^2 (population, ddof=0); std = sqrt(var + eps).
        operations += "$squared = stablehlo.multiply $grouped, $grouped : $groupedType"
        operations += "$sumSq = stablehlo.reduce($squared init: $zeroInit) " +
            "applies stablehlo.add across dimensions = [2] : ($groupedType, tensor<$elementType>) -> $reducedType"
        operations += "$meanSq = stablehlo.divide $sumSq, $countConst : $reducedType"
        operations += "$meanSquared = stablehlo.multiply $meanValue, $meanValue : $reducedType"
        operations += "$varValue = stablehlo.subtract $meanSq, $meanSquared : $reducedType"
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
     * Lower Upsample2d to traceable StableHLO. Input is NCHW (rank-4); scale,
     * mode and alignCorners are all static at trace time, so both modes lower
     * to fixed shape/linear ops (no runtime index math, no custom_call):
     *
     *  - Nearest: pixel replication via reshape -> broadcast_in_dim -> reshape,
     *    exactly matching the eager op out[oh,ow] = in[oh/sH, ow/sW].
     *
     *  - Bilinear: resize is a separable linear map, so we precompute the two
     *    resize matrices A_h [outH x inH] and A_w [outW x inW] (each row holds
     *    the two bilinear neighbor weights) as constants and apply them with two
     *    dot_general contractions. The weights are the same static floats the
     *    eager/numpy blend uses, so the result matches to fp tolerance.
     */
    private fun convertUpsample2d(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Upsample2d operation requires exactly 1 operand (input), got ${operands.size}",
                "Unsupported upsample2d arity for node ${node.id}"
            )
        }

        val typeMapper = context.getTypeMapper()
        val inputSpec = node.inputs.firstOrNull()
        val outputSpec = node.outputs.firstOrNull()
        val inputShape = inputSpec?.shape ?: outputSpec?.shape ?: emptyList()
        if (inputShape.size != 4) {
            return ConversionResult.Failure(
                "Upsample2d requires a rank-4 NCHW input, got rank ${inputShape.size}",
                "Unsupported upsample2d input rank for node ${node.id}"
            )
        }

        val params = node.operation.parameters
        val (scaleH, scaleW) = extractScalePair(params)
        if (scaleH < 1 || scaleW < 1) {
            return ConversionResult.Failure(
                "Upsample2d requires positive integer scale, got [$scaleH, $scaleW]",
                "Unsupported upsample2d scale for node ${node.id}"
            )
        }
        val mode = (params["mode"] as? String) ?: "Nearest"
        val alignCorners = (params["alignCorners"] as? Boolean) ?: false

        val n = inputShape[0]
        val c = inputShape[1]
        val h = inputShape[2]
        val w = inputShape[3]
        val outH = h * scaleH
        val outW = w * scaleW

        val elementType = inputSpec?.let { typeMapper.mapDType(it.dtype) }
            ?: outputSpec?.let { typeMapper.mapDType(it.dtype) }
            ?: "f32"
        val inputType = inputSpec?.let { typeMapper.mapTensorType(it) }
            ?: "tensor<${n}x${c}x${h}x${w}x$elementType>"
        val outputType = outputSpec?.let { typeMapper.mapTensorType(it) }
            ?: "tensor<${n}x${c}x${outH}x${outW}x$elementType>"

        val xInput = operands[0]
        val operations = mutableListOf<String>()

        return when (mode.lowercase()) {
            "nearest" -> {
                // Insert unit axes after H and W, replicate each pixel sH x sW, then
                // collapse (H,sH)->H*sH and (W,sW)->W*sW.
                val expandedType = "tensor<${n}x${c}x${h}x1x${w}x1x$elementType>"
                val replicatedType = "tensor<${n}x${c}x${h}x${scaleH}x${w}x${scaleW}x$elementType>"
                val expanded = context.nextTempValue()
                val replicated = context.nextTempValue()
                val result = context.nextTempValue()
                operations += "$expanded = stablehlo.reshape $xInput : ($inputType) -> $expandedType"
                operations += "$replicated = stablehlo.broadcast_in_dim $expanded, " +
                    "dims = [0, 1, 2, 3, 4, 5] : ($expandedType) -> $replicatedType"
                operations += "$result = stablehlo.reshape $replicated : ($replicatedType) -> $outputType"
                operations.forEach { context.emitOperation(it) }
                ConversionResult.Success(outputValueName = result, emittedOperations = operations)
            }

            "bilinear" -> {
                val ah = buildResizeMatrix(h, scaleH, alignCorners)   // [outH x inH]
                val aw = buildResizeMatrix(w, scaleW, alignCorners)   // [outW x inW]
                val ahType = "tensor<${outH}x${h}x$elementType>"
                val awType = "tensor<${outW}x${w}x$elementType>"
                // dot_general output layout = lhs-free ++ rhs-free, so contracting the
                // input H axis against A_h yields [N, C, inW, outH]; then contracting that
                // inW axis against A_w yields [N, C, outH, outW] — no transposes needed.
                val intermediateType = "tensor<${n}x${c}x${w}x${outH}x$elementType>"

                val ahConst = context.nextTempValue()
                val awConst = context.nextTempValue()
                val tmp = context.nextTempValue()
                val result = context.nextTempValue()

                operations += "$ahConst = stablehlo.constant dense<${denseMatrixLiteral(ah)}> : $ahType"
                operations += "$awConst = stablehlo.constant dense<${denseMatrixLiteral(aw)}> : $awType"
                operations += "$tmp = stablehlo.dot_general $xInput, $ahConst, " +
                    "contracting_dims = [2] x [1] : ($inputType, $ahType) -> $intermediateType"
                operations += "$result = stablehlo.dot_general $tmp, $awConst, " +
                    "contracting_dims = [2] x [1] : ($intermediateType, $awType) -> $outputType"
                operations.forEach { context.emitOperation(it) }
                ConversionResult.Success(outputValueName = result, emittedOperations = operations)
            }

            else -> ConversionResult.Failure(
                "Upsample2d mode '$mode' is not supported (expected Nearest or Bilinear)",
                "Unsupported upsample2d mode for node ${node.id}"
            )
        }
    }

    /** Read the [sH, sW] integer scale from op params (tape records List<Int>; also accept Pair/Int). */
    private fun extractScalePair(params: Map<String, Any>): Pair<Int, Int> {
        return when (val scale = params["scale"]) {
            is Pair<*, *> ->
                ((scale.first as? Number)?.toInt() ?: 1) to ((scale.second as? Number)?.toInt() ?: 1)
            is Number -> scale.toInt() to scale.toInt()
            is List<*> -> {
                val list = scale.mapNotNull { (it as? Number)?.toInt() }
                when {
                    list.size >= 2 -> list[0] to list[1]
                    list.size == 1 -> list[0] to list[0]
                    else -> 1 to 1
                }
            }
            else -> 1 to 1
        }
    }

    /**
     * Build the [outDim x inDim] bilinear resize matrix: row o holds the weights of
     * the (at most two) source neighbors for output index o, matching the eager
     * DefaultCpuOps coordinate map and border clamping. When the two neighbors clamp
     * to the same index their weights sum to 1.
     */
    private fun buildResizeMatrix(inDim: Int, scale: Int, alignCorners: Boolean): Array<FloatArray> {
        val outDim = inDim * scale
        val m = Array(outDim) { FloatArray(inDim) }
        for (o in 0 until outDim) {
            val src = if (alignCorners) {
                if (outDim <= 1) 0f else o.toFloat() * (inDim - 1) / (outDim - 1)
            } else {
                (o + 0.5f) / scale - 0.5f
            }
            val i0 = floor(src).toInt().coerceIn(0, inDim - 1)
            val i1 = (i0 + 1).coerceIn(0, inDim - 1)
            val frac = (src - i0).coerceIn(0.0f, 1.0f)
            m[o][i0] += (1f - frac)
            m[o][i1] += frac
        }
        return m
    }

    /** Render a 2D float matrix as a nested-bracket MLIR `dense<...>` literal. */
    private fun denseMatrixLiteral(m: Array<FloatArray>): String {
        return m.joinToString(prefix = "[", postfix = "]", separator = ", ") { row ->
            row.joinToString(prefix = "[", postfix = "]", separator = ", ") { v -> formatMlirFloat(v) }
        }
    }

    /** Float -> MLIR f-literal. Bilinear weights lie in [0,1] and render as plain decimals. */
    private fun formatMlirFloat(v: Float): String {
        val s = v.toString()
        return if (s.contains('.') || s.contains('e') || s.contains('E')) s else "$s.0"
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

        val axisSize = if (rank > 0) inputShape[axis] else 1
        val scaleType = "tensor<${axisSize}x$elementType>"

        val xSquared = context.nextTempValue()
        val zeroInit = context.nextTempValue()
        val countConst = context.nextTempValue()
        val sumSq = context.nextTempValue()
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

        // mean(x^2, axis) via real `stablehlo.reduce` (sum / count) — not the @reduce_mean
        // custom_call stub — so the module compiles on stock IREE (matching convertGroupNorm).
        operations += "$zeroInit = stablehlo.constant dense<0.0> : tensor<$elementType>"
        operations += "$countConst = stablehlo.constant dense<${axisSize}.0> : $reducedType"
        operations += "$sumSq = stablehlo.reduce($xSquared init: $zeroInit) " +
            "applies stablehlo.add across dimensions = [$axis] : ($outputType, tensor<$elementType>) -> $reducedType"
        operations += "$meanSquared = stablehlo.divide $sumSq, $countConst : $reducedType"

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

        // Final scale multiply is optional. Scale has shape [axisSize] and is broadcast over
        // the normalization axis; when the caller passed no scale we return normalized directly.
        val finalValue: String
        if (scaleOperand != null) {
            val scaleBroadcast = context.nextTempValue()
            operations += "$scaleBroadcast = stablehlo.broadcast_in_dim $scaleOperand, " +
                "dims = [$axis] : ($scaleType) -> $outputType"
            operations += "$resultValue = stablehlo.multiply $normalized, $scaleBroadcast : $outputType"
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