package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.compile.hlo.hasDynamic
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
        "sigmoid", "softmax", "tanh", "gelu", "swish",
        // SiLU (Sigmoid Linear Unit) is the name every Llama / Mistral /
        // Qwen / Gemma family model uses for the same x * sigmoid(x)
        // activation that PyTorch historically called swish. Register
        // the alias so traced LLM graphs don't fall through to the
        // "no converter found" path.
        "silu", "SiLU"
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
            "swish", "silu" -> convertSwish(node, operands, context)
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
            "tensor<${reducedShape.joinToString("x") { if (it < 0) "?" else "$it" }}x$elementType>"
        }
        // Dynamic softmax axis / leading dims (`?`): the reduced max/sum must broadcast back to a dynamic
        // output shape, which `stablehlo.broadcast_in_dim` cannot target — use `stablehlo.dynamic_broadcast_in_dim`
        // with a runtime `output_dimensions` operand built from the input via `get_dimension_size` (IREE's
        // stablehlo pipeline rejects CHLO implicit-broadcast ops as illegal).
        val dyn = inputShape.hasDynamic()
        val shapeType = "tensor<${rank}xi32>"

        // Dimensions kept for broadcast_in_dim: every input dim except `axis`,
        // mapped to its position in the reduced tensor.
        val broadcastDims = (0 until rank).filter { it != axis }.joinToString(", ")

        val maxInit = context.nextTempValue()
        val maxValue = context.nextTempValue()
        val maxBroadcast = context.nextTempValue()
        val shiftedValue = context.nextTempValue()
        val expValue = context.nextTempValue()
        val sumInit = context.nextTempValue()
        val sumValue = context.nextTempValue()
        val sumBroadcast = context.nextTempValue()
        val resultValue = context.nextTempValue()

        // Identity for stablehlo.maximum on floats: -inf. Spell it via the bit
        // pattern (width-matched to the element type — a 32-bit pattern in a bf16
        // constant is out of range).
        val maxIdentity = context.getTypeMapper().negInfBits(elementType)

        val operations = buildList {
            // Reduce-max along the softmax axis (for numerical stability).
            add("$maxInit = stablehlo.constant dense<$maxIdentity> : tensor<$elementType>")
            add(
                "$maxValue = stablehlo.reduce(${operands[0]} init: $maxInit) " +
                    "applies stablehlo.maximum across dimensions = [$axis] : " +
                    "($outputType, tensor<$elementType>) -> $reducedType",
            )
            // Build the runtime output-shape operand once (only needed for dynamic broadcasts).
            val shapeOperand: String = if (!dyn) "" else run {
                val parts = inputShape.indices.map { d ->
                    if (inputShape[d] >= 0) {
                        val c = context.nextTempValue()
                        add("$c = stablehlo.constant dense<${inputShape[d]}> : tensor<1xi32>")
                        c
                    } else {
                        val gd = context.nextTempValue(); val gr = context.nextTempValue()
                        add("$gd = stablehlo.get_dimension_size ${operands[0]}, dim = $d : ($outputType) -> tensor<i32>")
                        add("$gr = stablehlo.reshape $gd : (tensor<i32>) -> tensor<1xi32>")
                        gr
                    }
                }
                val sh = context.nextTempValue()
                add("$sh = stablehlo.concatenate ${parts.joinToString(", ")}, dim = 0 : (${parts.joinToString(", ") { "tensor<1xi32>" }}) -> $shapeType")
                sh
            }
            // Subtract the max: static → explicit broadcast_in_dim; dynamic → runtime dynamic_broadcast_in_dim.
            if (dyn) {
                add("$maxBroadcast = stablehlo.dynamic_broadcast_in_dim $maxValue, $shapeOperand, dims = [$broadcastDims] : ($reducedType, $shapeType) -> $outputType")
            } else {
                add("$maxBroadcast = stablehlo.broadcast_in_dim $maxValue, dims = [$broadcastDims] : ($reducedType) -> $outputType")
            }
            add("$shiftedValue = stablehlo.subtract ${operands[0]}, $maxBroadcast : $outputType")
            add("$expValue = stablehlo.exponential $shiftedValue : $outputType")
            add("$sumInit = stablehlo.constant dense<0.0> : tensor<$elementType>")
            add(
                "$sumValue = stablehlo.reduce($expValue init: $sumInit) " +
                    "applies stablehlo.add across dimensions = [$axis] : " +
                    "($outputType, tensor<$elementType>) -> $reducedType",
            )
            // Normalize: static → broadcast_in_dim; dynamic → runtime dynamic_broadcast_in_dim.
            if (dyn) {
                add("$sumBroadcast = stablehlo.dynamic_broadcast_in_dim $sumValue, $shapeOperand, dims = [$broadcastDims] : ($reducedType, $shapeType) -> $outputType")
            } else {
                add("$sumBroadcast = stablehlo.broadcast_in_dim $sumValue, dims = [$broadcastDims] : ($reducedType) -> $outputType")
            }
            add("$resultValue = stablehlo.divide $expValue, $sumBroadcast : $outputType")
        }

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