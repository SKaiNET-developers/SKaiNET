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
        "layerNorm", "layerNormalization", "LayerNormalization"
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

        val convOperation = buildConv1dOperation(
            resultValue = resultValue,
            input = operands[0],
            weight = operands[1],
            bias = if (operands.size > 2) operands[2] else null,
            inputType = inputType,
            weightType = weightType,
            outputType = outputType,
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

        // Build StableHLO convolution operation
        val convOperation = buildConvolutionOperation(
            resultValue = resultValue,
            input = operands[0],
            weight = operands[1],
            bias = if (operands.size > 2) operands[2] else null,
            inputType = inputType,
            weightType = weightType,
            outputType = outputType,
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
    
    private fun convertLayerNorm(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 1) {
            return ConversionResult.Failure(
                "LayerNorm operation requires at least 1 operand (input), got ${operands.size}",
                "Unsupported layerNorm arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?xf32>"
        
        // Extract layer norm parameters
        val params = node.operation.parameters
        val epsilon = params["eps"] as? Double ?: 1e-5
        val normalizedShape = params["normalized_shape"] as? IntArray ?: intArrayOf(-1)
        
        val resultValue = context.nextTempValue()
        
        // Build StableHLO layer normalization using reduce operations
        val layerNormOperation = buildLayerNormOperation(
            resultValue = resultValue,
            input = operands[0],
            scale = if (operands.size > 1) operands[1] else null,
            offset = if (operands.size > 2) operands[2] else null,
            outputType = outputType,
            epsilon = epsilon,
            normalizedShape = normalizedShape
        )
        
        context.emitOperation(layerNormOperation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(layerNormOperation)
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
            val convOp = convCore(convResult)
            "$convOp\n    $resultValue = stablehlo.add $convResult, $bias : $outputType"
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
            val convOp = convCore(convResult)
            "$convOp\n    $resultValue = stablehlo.add $convResult, $bias : $outputType"
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
    
    private fun buildLayerNormOperation(
        resultValue: String,
        input: String,
        scale: String?,
        offset: String?,
        outputType: String,
        epsilon: Double,
        normalizedShape: IntArray
    ): String {
        // Layer normalization is implemented using reduce operations
        // This is a simplified version - full implementation would need proper mean/variance computation
        return if (scale != null && offset != null) {
            "$resultValue = stablehlo.custom_call @layer_norm($input, $scale, $offset) " +
                    "{epsilon = $epsilon} : $outputType"
        } else {
            "$resultValue = stablehlo.custom_call @layer_norm($input) " +
                    "{epsilon = $epsilon} : $outputType"
        }
    }
}