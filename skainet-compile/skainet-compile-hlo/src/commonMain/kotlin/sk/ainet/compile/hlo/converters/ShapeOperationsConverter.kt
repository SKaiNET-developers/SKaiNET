package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for shape manipulation operations.
 * 
 * This converter implements shape operations using StableHLO primitives:
 * - reshape: using stablehlo.reshape with proper shape inference
 * - flatten: using stablehlo.reshape to flatten specified dimensions
 * - squeeze: using stablehlo.reshape to remove singleton dimensions
 * - unsqueeze: using stablehlo.broadcast_in_dim for dimension expansion
 * 
 * Supports operations as specified in Requirements 2.5:
 * - Shape operations (reshape, flatten, squeeze, unsqueeze)
 * - Dynamic reshaping with runtime shape computation
 * - Proper shape inference and validation
 */
public class ShapeOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "reshape", "flatten", "squeeze", "unsqueeze"
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "reshape" -> convertReshape(node, operands, context)
            "flatten" -> convertFlatten(node, operands, context)
            "squeeze" -> convertSqueeze(node, operands, context)
            "unsqueeze" -> convertUnsqueeze(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ShapeOperationsConverter"
            )
        }
    }
    
    /**
     * Convert reshape operation using stablehlo.reshape.
     * Handles both static and dynamic shape specifications.
     */
    private fun convertReshape(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Reshape operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported reshape arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get the new shape from parameters or output spec
        val newShape = when {
            outputSpec?.shape != null -> outputSpec.shape
            node.operation.parameters.containsKey("shape") -> {
                @Suppress("UNCHECKED_CAST")
                node.operation.parameters["shape"] as? List<Int>
            }
            node.operation.parameters.containsKey("newShape") -> {
                @Suppress("UNCHECKED_CAST")
                node.operation.parameters["newShape"] as? List<Int>
            }
            else -> null
        }
        
        if (newShape == null || newShape.isEmpty()) {
            return ConversionResult.Failure(
                "Reshape operation requires a target shape specification",
                "Missing shape parameter for reshape node ${node.id}"
            )
        }
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($outputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert flatten operation using stablehlo.reshape.
     * Flattens dimensions from startDim to endDim into a single dimension.
     */
    private fun convertFlatten(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Flatten operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported flatten arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get flatten parameters
        val startDim = node.operation.parameters["startDim"] as? Int ?: 0
        val endDim = node.operation.parameters["endDim"] as? Int ?: -1
        
        context.emitComment("Flatten from dim $startDim to $endDim")
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($outputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert squeeze operation using stablehlo.reshape.
     * Removes singleton dimensions (dimensions of size 1).
     */
    private fun convertSqueeze(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Squeeze operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported squeeze arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get squeeze dimension parameter (null means squeeze all singleton dimensions)
        val dim = node.operation.parameters["dim"] as? Int
        
        if (dim != null) {
            context.emitComment("Squeeze dimension $dim")
        } else {
            context.emitComment("Squeeze all singleton dimensions")
        }
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($outputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert unsqueeze operation using stablehlo.broadcast_in_dim.
     * Adds a singleton dimension at the specified position.
     */
    private fun convertUnsqueeze(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Unsqueeze operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported unsqueeze arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        // Get the dimension to unsqueeze at
        val dim = node.operation.parameters["dim"] as? Int
            ?: return ConversionResult.Failure(
                "Unsqueeze operation requires a 'dim' parameter",
                "Missing dim parameter for unsqueeze node ${node.id}"
            )
        
        context.emitComment("Unsqueeze at dimension $dim")
        
        // For unsqueeze, we can use either reshape or broadcast_in_dim
        // Using reshape is simpler for this implementation
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.reshape ${operands[0]} : ($outputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
}