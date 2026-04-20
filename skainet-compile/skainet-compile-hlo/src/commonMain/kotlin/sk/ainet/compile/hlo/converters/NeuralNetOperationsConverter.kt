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
        "rmsNorm", "rms_norm", "RMSNorm", "RmsNorm"
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
        
        // Build StableHLO reduce_window operation for max pooling
        val operations = buildMaxPoolOperations(
            resultValue = resultValue,
            input = operands[0],
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
        
        // Build StableHLO reduce_window operation for average pooling
        val operations = buildAvgPoolOperations(
            resultValue = resultValue,
            input = operands[0],
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

    private fun buildMaxPoolOperations(
        resultValue: String,
        input: String,
        outputType: String,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        context: ConversionContext
    ): List<String> {
        // For max pooling, we need to create a negative infinity constant as the initial value
        val initValue = context.nextTempValue()
        val initConstant = "$initValue = stablehlo.constant dense<-3.4028235e+38> : tensor<f32>"
        
        val poolOp = "$resultValue = stablehlo.reduce_window($input, $initValue) " +
                "applies stablehlo.maximum " +
                "over window dimensions = [${kernelSize.first}, ${kernelSize.second}] " +
                "stride = [${stride.first}, ${stride.second}] " +
                "pad = [[${padding.first}, ${padding.first}], [${padding.second}, ${padding.second}]] : $outputType"
        
        // Emit operations through context
        context.emitOperation(initConstant)
        context.emitOperation(poolOp)
        
        return listOf(initConstant, poolOp)
    }
    
    private fun buildAvgPoolOperations(
        resultValue: String,
        input: String,
        outputType: String,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        context: ConversionContext
    ): List<String> {
        // Average pooling requires sum + division by kernel size
        val kernelArea = kernelSize.first * kernelSize.second
        val initZero = context.nextTempValue()
        val kernelAreaConst = context.nextTempValue()
        val sumResult = context.nextTempValue()
        
        val initConstant = "$initZero = stablehlo.constant dense<0.0> : tensor<f32>"
        val areaConstant = "$kernelAreaConst = stablehlo.constant dense<$kernelArea.0> : tensor<f32>"
        
        val sumOp = "$sumResult = stablehlo.reduce_window($input, $initZero) " +
                "applies stablehlo.add " +
                "over window dimensions = [${kernelSize.first}, ${kernelSize.second}] " +
                "stride = [${stride.first}, ${stride.second}] " +
                "pad = [[${padding.first}, ${padding.first}], [${padding.second}, ${padding.second}]] : $outputType"
        
        val divideOp = "$resultValue = stablehlo.divide $sumResult, $kernelAreaConst : $outputType"
        
        // Emit operations through context
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
    
}