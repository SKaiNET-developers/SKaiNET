package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for activation function operations.
 * 
 * This converter implements various activation functions using StableHLO primitives:
 * - sigmoid: using stablehlo.exponential and arithmetic operations
 * - softmax: using stablehlo.reduce and stablehlo.broadcast_in_dim
 * - tanh, gelu, swish: using appropriate StableHLO operations
 * 
 * Note: relu is already implemented in LegacyOperationsConverter
 * 
 * Supports operations as specified in Requirements 2.3:
 * - Activation functions (relu, sigmoid, softmax)
 * - Additional activations (tanh, gelu, swish)
 */
public class ActivationOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "sigmoid", "softmax", "tanh", "gelu", "swish"
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "sigmoid" -> convertSigmoid(node, operands, context)
            "softmax" -> convertSoftmax(node, operands, context)
            "tanh" -> convertTanh(node, operands, context)
            "gelu" -> convertGelu(node, operands, context)
            "swish" -> convertSwish(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ActivationOperationsConverter"
            )
        }
    }
    
    /**
     * Convert sigmoid activation using stablehlo.exponential and arithmetic operations.
     * sigmoid(x) = 1 / (1 + exp(-x))
     */
    private fun convertSigmoid(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Sigmoid operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported sigmoid arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val negatedValue = context.nextTempValue()
        val expValue = context.nextTempValue()
        val oneConstValue = context.nextTempValue()
        val onePlusExpValue = context.nextTempValue()
        val resultValue = context.nextTempValue()
        
        val operations = listOf(
            "$negatedValue = stablehlo.negate ${operands[0]} : $outputType",
            "$expValue = stablehlo.exponential $negatedValue : $outputType",
            "$oneConstValue = stablehlo.constant dense<1.0> : $outputType",
            "$onePlusExpValue = stablehlo.add $oneConstValue, $expValue : $outputType",
            "$resultValue = stablehlo.divide $oneConstValue, $onePlusExpValue : $outputType"
        )
        
        operations.forEach { context.emitOperation(it) }
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
    
    /**
     * Convert softmax activation using real reductions and broadcast_in_dim.
     * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
     *
     * The max and sum terms are lowered to stablehlo.custom_call @reduce_max
     * and @reduce_sum — matching ReductionOperationsConverter's style — then
     * broadcast back to the input shape before subtract / divide. This replaces
     * an earlier lowering that used `dense<0.0>` / `dense<1.0>` placeholder
     * constants (see #467) and produced numerically wrong MLIR.
     */
    private fun convertSoftmax(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Softmax operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported softmax arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) }
            ?: "f32"

        val inputShape = node.inputs.firstOrNull()?.shape ?: outputSpec?.shape ?: emptyList()
        val rank = inputShape.size

        // Normalize axis against rank. Default to the last dimension.
        val rawAxis = node.operation.parameters["axis"] as? Int ?: -1
        val axis = when {
            rank == 0 -> 0
            rawAxis < 0 -> rank + rawAxis
            else -> rawAxis
        }.coerceIn(0, (rank - 1).coerceAtLeast(0))

        // Reduced tensor type: input shape with `axis` dimension removed.
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

        // Dimensions kept for broadcast_in_dim: every input dim except `axis`,
        // mapped to its position in the reduced tensor.
        val broadcastDims = (0 until rank).filter { it != axis }.joinToString(", ")

        val maxValue = context.nextTempValue()
        val maxBroadcast = context.nextTempValue()
        val shiftedValue = context.nextTempValue()
        val expValue = context.nextTempValue()
        val sumValue = context.nextTempValue()
        val sumBroadcast = context.nextTempValue()
        val resultValue = context.nextTempValue()

        val operations = listOf(
            // Reduce-max along the softmax axis (for numerical stability).
            "$maxValue = stablehlo.custom_call @reduce_max(${operands[0]}) " +
                "{dimensions = [$axis], keepdim = false} : $reducedType",

            // Broadcast reduced max back to the input shape.
            "$maxBroadcast = stablehlo.broadcast_in_dim $maxValue, " +
                "dims = [$broadcastDims] : ($reducedType) -> $outputType",

            // Subtract the max for numerical stability.
            "$shiftedValue = stablehlo.subtract ${operands[0]}, $maxBroadcast : $outputType",

            // Elementwise exponential.
            "$expValue = stablehlo.exponential $shiftedValue : $outputType",

            // Reduce-sum along the softmax axis.
            "$sumValue = stablehlo.custom_call @reduce_sum($expValue) " +
                "{dimensions = [$axis], keepdim = false} : $reducedType",

            // Broadcast the sum back to the input shape.
            "$sumBroadcast = stablehlo.broadcast_in_dim $sumValue, " +
                "dims = [$broadcastDims] : ($reducedType) -> $outputType",

            // Normalize.
            "$resultValue = stablehlo.divide $expValue, $sumBroadcast : $outputType"
        )

        operations.forEach { context.emitOperation(it) }

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
    
    /**
     * Convert tanh activation using stablehlo.tanh.
     */
    private fun convertTanh(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Tanh operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported tanh arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.tanh ${operands[0]} : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert GELU activation using approximation.
     * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     */
    private fun convertGelu(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "GELU operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported gelu arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val halfConst = context.nextTempValue()
        val oneConst = context.nextTempValue()
        val coeff1Const = context.nextTempValue()  // sqrt(2/π) ≈ 0.7978845608
        val coeff2Const = context.nextTempValue()  // 0.044715
        val xSquared = context.nextTempValue()
        val xCubed = context.nextTempValue()
        val cubicTerm = context.nextTempValue()
        val innerSum = context.nextTempValue()
        val scaledSum = context.nextTempValue()
        val tanhValue = context.nextTempValue()
        val onePlusTanh = context.nextTempValue()
        val halfX = context.nextTempValue()
        val resultValue = context.nextTempValue()
        
        val operations = listOf(
            "$halfConst = stablehlo.constant dense<0.5> : $outputType",
            "$oneConst = stablehlo.constant dense<1.0> : $outputType",
            "$coeff1Const = stablehlo.constant dense<0.7978845608> : $outputType",
            "$coeff2Const = stablehlo.constant dense<0.044715> : $outputType",
            "$xSquared = stablehlo.multiply ${operands[0]}, ${operands[0]} : $outputType",
            "$xCubed = stablehlo.multiply $xSquared, ${operands[0]} : $outputType",
            "$cubicTerm = stablehlo.multiply $coeff2Const, $xCubed : $outputType",
            "$innerSum = stablehlo.add ${operands[0]}, $cubicTerm : $outputType",
            "$scaledSum = stablehlo.multiply $coeff1Const, $innerSum : $outputType",
            "$tanhValue = stablehlo.tanh $scaledSum : $outputType",
            "$onePlusTanh = stablehlo.add $oneConst, $tanhValue : $outputType",
            "$halfX = stablehlo.multiply $halfConst, ${operands[0]} : $outputType",
            "$resultValue = stablehlo.multiply $halfX, $onePlusTanh : $outputType"
        )
        
        operations.forEach { context.emitOperation(it) }
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
    
    /**
     * Convert Swish activation (also known as SiLU).
     * Swish(x) = x * sigmoid(x)
     */
    private fun convertSwish(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Swish operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported swish arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // First compute sigmoid(x)
        val negatedValue = context.nextTempValue()
        val expValue = context.nextTempValue()
        val oneConstValue = context.nextTempValue()
        val onePlusExpValue = context.nextTempValue()
        val sigmoidValue = context.nextTempValue()
        val resultValue = context.nextTempValue()
        
        val operations = listOf(
            // Compute sigmoid(x)
            "$negatedValue = stablehlo.negate ${operands[0]} : $outputType",
            "$expValue = stablehlo.exponential $negatedValue : $outputType",
            "$oneConstValue = stablehlo.constant dense<1.0> : $outputType",
            "$onePlusExpValue = stablehlo.add $oneConstValue, $expValue : $outputType",
            "$sigmoidValue = stablehlo.divide $oneConstValue, $onePlusExpValue : $outputType",
            // Multiply x * sigmoid(x)
            "$resultValue = stablehlo.multiply ${operands[0]}, $sigmoidValue : $outputType"
        )
        
        operations.forEach { context.emitOperation(it) }
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = operations
        )
    }
}