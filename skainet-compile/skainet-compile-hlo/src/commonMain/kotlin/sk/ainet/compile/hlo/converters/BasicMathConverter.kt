package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for basic mathematical operations (add, subtract, multiply, divide).
 * 
 * This converter handles the basic arithmetic operations by mapping them
 * to their corresponding StableHLO operations. It supports:
 * - Element-wise operations with broadcasting
 * - Mixed-type arithmetic with automatic type promotion
 * - Proper operand ordering and type consistency
 */
public class BasicMathConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "add", "subtract", "multiply", "divide",
        "sub", "mul", "div" // Common aliases
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Math operations require exactly 2 operands, got ${operands.size}",
                "Unsupported ${node.operation.name} arity for node ${node.id}"
            )
        }
        
        return try {
            convertMathOperation(node, operands, context)
        } catch (e: Exception) {
            ConversionResult.Failure(
                "Error converting ${node.operation.name}: ${e.message}",
                "Failed to convert math operation for node ${node.id}"
            )
        }
    }
    
    private fun convertMathOperation(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        val inputNodes = context.getInputNodes(node)
        if (inputNodes.size != 2) {
            return ConversionResult.Failure(
                "Cannot determine input types for math operation",
                "Missing input node information for ${node.id}"
            )
        }
        
        val leftSpec = inputNodes[0].outputs.firstOrNull()
        val rightSpec = inputNodes[1].outputs.firstOrNull()
        
        if (leftSpec == null || rightSpec == null) {
            return ConversionResult.Failure(
                "Cannot determine input tensor specifications",
                "Missing tensor specs for math operation ${node.id}"
            )
        }
        
        // Handle type promotion and broadcasting
        val (promotedOperands, resultType) = handleTypePromotionAndBroadcasting(
            operands, leftSpec, rightSpec, context
        )
        
        val resultValue = context.nextTempValue()
        val stableHloOp = getStableHloOperation(node.operation.name)
            ?: return ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by BasicMathConverter"
            )
        
        val operation = "$resultValue = $stableHloOp ${promotedOperands[0]}, ${promotedOperands[1]} : $resultType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    private fun getStableHloOperation(operationName: String): String? {
        return when (operationName.lowercase()) {
            "add" -> "stablehlo.add"
            "subtract", "sub" -> "stablehlo.subtract"
            "multiply", "mul" -> "stablehlo.multiply"
            "divide", "div" -> "stablehlo.divide"
            else -> null
        }
    }
    
    private fun handleTypePromotionAndBroadcasting(
        operands: List<String>,
        leftSpec: sk.ainet.lang.tensor.ops.TensorSpec,
        rightSpec: sk.ainet.lang.tensor.ops.TensorSpec,
        context: ConversionContext
    ): Pair<List<String>, String> {
        val typeMapper = context.getTypeMapper()
        
        // Check if types are already compatible
        if (typeMapper.areTypesCompatible(leftSpec, rightSpec)) {
            val outputType = typeMapper.mapTensorType(leftSpec)
            return Pair(operands, outputType)
        }
        
        // Perform type promotion
        val promotedSpec = typeMapper.inferBroadcastType(leftSpec, rightSpec)
        val resultType = typeMapper.mapTensorType(promotedSpec)
        
        // Check if we need explicit broadcasting or type conversion
        val promotedOperands = mutableListOf<String>()
        
        // Handle left operand
        if (!isSameType(leftSpec, promotedSpec, typeMapper)) {
            val convertedLeft = context.nextTempValue()
            val leftTargetType = typeMapper.mapTensorType(promotedSpec)
            context.emitOperation("$convertedLeft = stablehlo.convert ${operands[0]} : ${typeMapper.mapTensorType(leftSpec)} -> $leftTargetType")
            promotedOperands.add(convertedLeft)
        } else {
            promotedOperands.add(operands[0])
        }
        
        // Handle right operand
        if (!isSameType(rightSpec, promotedSpec, typeMapper)) {
            val convertedRight = context.nextTempValue()
            val rightTargetType = typeMapper.mapTensorType(promotedSpec)
            context.emitOperation("$convertedRight = stablehlo.convert ${operands[1]} : ${typeMapper.mapTensorType(rightSpec)} -> $rightTargetType")
            promotedOperands.add(convertedRight)
        } else {
            promotedOperands.add(operands[1])
        }
        
        // Handle broadcasting if shapes are different
        val finalOperands = handleBroadcasting(promotedOperands, leftSpec, rightSpec, promotedSpec, context)
        
        return Pair(finalOperands, resultType)
    }
    
    private fun handleBroadcasting(
        operands: List<String>,
        leftSpec: sk.ainet.lang.tensor.ops.TensorSpec,
        rightSpec: sk.ainet.lang.tensor.ops.TensorSpec,
        targetSpec: sk.ainet.lang.tensor.ops.TensorSpec,
        context: ConversionContext
    ): List<String> {
        val typeMapper = context.getTypeMapper()
        val finalOperands = mutableListOf<String>()
        
        // Check if left operand needs broadcasting
        if (!isSameShape(leftSpec, targetSpec)) {
            val broadcastLeft = context.nextTempValue()
            val targetType = typeMapper.mapTensorType(targetSpec)
            val broadcastDims = computeBroadcastDims(leftSpec.shape, targetSpec.shape)
            context.emitOperation("$broadcastLeft = stablehlo.broadcast_in_dim ${operands[0]} : ${typeMapper.mapTensorType(leftSpec)} -> $targetType")
            context.emitOperation("  broadcast_dimensions = [$broadcastDims]")
            finalOperands.add(broadcastLeft)
        } else {
            finalOperands.add(operands[0])
        }
        
        // Check if right operand needs broadcasting
        if (!isSameShape(rightSpec, targetSpec)) {
            val broadcastRight = context.nextTempValue()
            val targetType = typeMapper.mapTensorType(targetSpec)
            val broadcastDims = computeBroadcastDims(rightSpec.shape, targetSpec.shape)
            context.emitOperation("$broadcastRight = stablehlo.broadcast_in_dim ${operands[1]} : ${typeMapper.mapTensorType(rightSpec)} -> $targetType")
            context.emitOperation("  broadcast_dimensions = [$broadcastDims]")
            finalOperands.add(broadcastRight)
        } else {
            finalOperands.add(operands[1])
        }
        
        return finalOperands
    }
    
    private fun isSameType(
        spec1: sk.ainet.lang.tensor.ops.TensorSpec,
        spec2: sk.ainet.lang.tensor.ops.TensorSpec,
        typeMapper: sk.ainet.compile.hlo.TypeMapper
    ): Boolean {
        return typeMapper.mapDType(spec1.dtype) == typeMapper.mapDType(spec2.dtype)
    }
    
    private fun isSameShape(
        spec1: sk.ainet.lang.tensor.ops.TensorSpec,
        spec2: sk.ainet.lang.tensor.ops.TensorSpec
    ): Boolean {
        return spec1.shape == spec2.shape
    }
    
    private fun computeBroadcastDims(sourceShape: List<Int>?, targetShape: List<Int>?): String {
        if (sourceShape == null || targetShape == null) {
            return ""
        }
        
        // Simple broadcast dimension computation
        // For now, assume the source dimensions align with the last dimensions of target
        val sourceDims = sourceShape.size
        val targetDims = targetShape.size
        
        if (sourceDims == targetDims) {
            return (0 until sourceDims).joinToString(", ")
        }
        
        // Broadcast from smaller to larger shape
        val offset = targetDims - sourceDims
        return (offset until targetDims).joinToString(", ")
    }
}