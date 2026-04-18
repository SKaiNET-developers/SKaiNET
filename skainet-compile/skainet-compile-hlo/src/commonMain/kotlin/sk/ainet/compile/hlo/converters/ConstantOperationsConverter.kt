package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Converter for constant value operations in StableHLO.
 * 
 * This converter handles the generation of stablehlo.constant operations for various
 * types of constant values including scalars, tensors, splat values, and parameter
 * tensors. It supports constant folding opportunities during conversion and handles
 * learned weights as constants.
 * 
 * Supports operations as specified in Requirements 4.2:
 * - Scalar constants (single values)
 * - Dense tensor constants (multi-dimensional arrays)
 * - Splat constants (single value broadcasted to tensor shape)
 * - Parameter tensors and learned weights as constants
 * - Constant folding opportunities during conversion
 */
public class ConstantOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        // Basic constant operations
        "constant", "const",
        // Scalar constants
        "scalar_constant", "scalar",
        // Tensor constants
        "tensor_constant", "dense_constant",
        // Splat constants (single value broadcasted)
        "splat_constant", "splat", "fill",
        // Parameter and weight constants
        "parameter", "param", "weight", "bias",
        // Zero and one constants
        "zeros", "ones", "zeros_like", "ones_like"
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "constant", "const" -> convertGenericConstant(node, operands, context)
            "scalar_constant", "scalar" -> convertScalarConstant(node, operands, context)
            "tensor_constant", "dense_constant" -> convertTensorConstant(node, operands, context)
            "splat_constant", "splat", "fill" -> convertSplatConstant(node, operands, context)
            "parameter", "param", "weight", "bias" -> convertParameterConstant(node, operands, context)
            "zeros" -> convertZerosConstant(node, operands, context)
            "ones" -> convertOnesConstant(node, operands, context)
            "zeros_like", "ones_like" -> convertLikeConstant(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ConstantOperationsConverter"
            )
        }
    }
    
    /**
     * Convert a generic constant operation by inferring the type from parameters
     */
    private fun convertGenericConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        val value = node.operation.parameters["value"]
        val shape = node.operation.parameters["shape"] as? List<*>
        
        return when {
            value is Number && (shape == null || shape.isEmpty()) -> 
                convertScalarConstant(node, operands, context)
            value is Number && shape != null -> 
                convertSplatConstant(node, operands, context)
            value is List<*> -> 
                convertTensorConstant(node, operands, context)
            else -> ConversionResult.Failure(
                "Unsupported constant value type: ${value?.let { it::class.simpleName }}",
                "Cannot determine constant type for node ${node.id}"
            )
        }
    }
    
    /**
     * Convert a scalar constant to stablehlo.constant
     */
    private fun convertScalarConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        // Scalar constants should not have operands
        if (operands.isNotEmpty()) {
            return ConversionResult.Failure(
                "Scalar constant should not have operands, got ${operands.size}",
                "Invalid scalar constant for node ${node.id}"
            )
        }
        
        val value = node.operation.parameters["value"] as? Number
            ?: return ConversionResult.Failure(
                "Missing or invalid 'value' parameter for scalar constant",
                "No value specified for scalar constant ${node.id}"
            )
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<f32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.constant dense<${formatConstantValue(value)}> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert a tensor constant with dense values
     */
    private fun convertTensorConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        // Tensor constants should not have operands
        if (operands.isNotEmpty()) {
            return ConversionResult.Failure(
                "Tensor constant should not have operands, got ${operands.size}",
                "Invalid tensor constant for node ${node.id}"
            )
        }
        
        val values = node.operation.parameters["values"] as? List<*>
            ?: node.operation.parameters["data"] as? List<*>
            ?: return ConversionResult.Failure(
                "Missing 'values' or 'data' parameter for tensor constant",
                "No data specified for tensor constant ${node.id}"
            )
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val formattedValues = formatTensorValues(values, outputSpec)
        val operation = "$resultValue = stablehlo.constant dense<$formattedValues> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert a splat constant (single value broadcasted to tensor shape)
     */
    private fun convertSplatConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        // Splat constants should not have operands
        if (operands.isNotEmpty()) {
            return ConversionResult.Failure(
                "Splat constant should not have operands, got ${operands.size}",
                "Invalid splat constant for node ${node.id}"
            )
        }
        
        val value = node.operation.parameters["value"] as? Number
            ?: return ConversionResult.Failure(
                "Missing or invalid 'value' parameter for splat constant",
                "No value specified for splat constant ${node.id}"
            )
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.constant dense<${formatConstantValue(value)}> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert parameter tensors and learned weights as constants
     */
    private fun convertParameterConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        // Parameters can have initial values or be uninitialized
        val initialValue = node.operation.parameters["initial_value"]
        val isTrainable = node.operation.parameters["trainable"] as? Boolean ?: true
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        
        // Add comment about parameter nature
        val paramType = if (isTrainable) "trainable parameter" else "frozen parameter"
        context.emitComment("${node.operation.name} ${node.id}: $paramType")
        
        val operation = when {
            initialValue is Number -> {
                "$resultValue = stablehlo.constant dense<${formatConstantValue(initialValue)}> : $outputType"
            }
            initialValue is List<*> -> {
                val formattedValues = formatTensorValues(initialValue, outputSpec)
                "$resultValue = stablehlo.constant dense<$formattedValues> : $outputType"
            }
            else -> {
                // Default initialization (zeros for now, could be configurable)
                context.emitComment("Using default zero initialization for parameter ${node.id}")
                "$resultValue = stablehlo.constant dense<0.0> : $outputType"
            }
        }
        
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert zeros constant
     */
    private fun convertZerosConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isNotEmpty()) {
            return ConversionResult.Failure(
                "Zeros constant should not have operands, got ${operands.size}",
                "Invalid zeros constant for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.constant dense<0.0> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert ones constant
     */
    private fun convertOnesConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.isNotEmpty()) {
            return ConversionResult.Failure(
                "Ones constant should not have operands, got ${operands.size}",
                "Invalid ones constant for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.constant dense<1.0> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert zeros_like or ones_like constants (shape inferred from operand)
     */
    private fun convertLikeConstant(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "${node.operation.name} requires exactly 1 operand, got ${operands.size}",
                "Invalid ${node.operation.name} for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val value = if (node.operation.name.lowercase().startsWith("zeros")) "0.0" else "1.0"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.constant dense<$value> : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Format a constant value for MLIR output
     */
    private fun formatConstantValue(value: Number): String {
        return when (value) {
            is Float -> if (value.isFinite()) value.toString() else "0.0"
            is Double -> if (value.isFinite()) value.toString() else "0.0"
            is Int -> value.toString() + ".0" // Convert to float format
            is Long -> value.toString() + ".0"
            else -> value.toString()
        }
    }
    
    /**
     * Format tensor values for MLIR dense constant.
     * MLIR dense<> syntax requires nested brackets matching the tensor rank:
     *   scalar:  dense<42.0>
     *   1D [3]:  dense<[v0, v1, v2]>
     *   2D [2,3]: dense<[[v0,v1,v2],[v3,v4,v5]]>
     *   4D [1,3,1,1]: dense<[[[[v0],[v1],[v2]]]]>
     *
     * Splat collapse: when every element is the same value and the input
     * list fully covers the shape, emit the single-scalar splat form
     * (`dense<v>` ≡ `dense<[[v, v, ...], ...]>` for any rank). This is the
     * first-pass lever against the 151 MB MLIR-text blowup described in
     * #519 — uninitialized VoidTensorOps-backed weights are uniform by
     * construction and compress from O(N*M) characters down to one.
     */
    private fun formatTensorValues(values: List<*>, outputSpec: TensorSpec?): String {
        val shape = outputSpec?.shape ?: emptyList()
        val expectedSize = if (shape.isEmpty()) values.size else shape.fold(1) { acc, d -> acc * d }
        if (values.isNotEmpty() && values.size >= expectedSize && values.toSet().size == 1) {
            return formatConstantValue(values[0] as Number)
        }

        return when {
            values.isEmpty() -> "0.0"
            values.size == 1 -> formatConstantValue(values[0] as Number)
            shape.isEmpty() -> "[" + values.joinToString(", ") { formatConstantValue(it as Number) } + "]"
            else -> formatNestedTensor(values, shape, 0, IntArray(1))
        }
    }

    /**
     * Recursively format a flat list of values into nested MLIR dense literal
     * matching the given shape. [offset] tracks the current position in the flat values list.
     */
    private fun formatNestedTensor(values: List<*>, shape: List<Int>, dim: Int, offset: IntArray): String {
        if (dim == shape.size - 1) {
            // Innermost dimension: emit a flat array of values
            val size = shape[dim]
            val sb = StringBuilder("[")
            for (i in 0 until size) {
                if (i > 0) sb.append(", ")
                val idx = offset[0]++
                if (idx < values.size) {
                    sb.append(formatConstantValue(values[idx] as Number))
                } else {
                    sb.append("0.0")
                }
            }
            sb.append("]")
            return sb.toString()
        }

        // Non-innermost dimension: recurse
        val size = shape[dim]
        val sb = StringBuilder("[")
        for (i in 0 until size) {
            if (i > 0) sb.append(", ")
            sb.append(formatNestedTensor(values, shape, dim + 1, offset))
        }
        sb.append("]")
        return sb.toString()
    }
}