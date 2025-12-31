package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for legacy operations that were supported in the original implementation.
 * 
 * This converter maintains compatibility with the existing add, matmul, and relu operations
 * while using the new modular architecture.
 */
public class LegacyOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "add", "matmul", "relu"
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "add" -> convertAdd(node, operands, context)
            "matmul" -> convertMatmul(node, operands, context)
            "relu" -> convertRelu(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by LegacyOperationsConverter"
            )
        }
    }
    
    private fun convertAdd(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Add operation requires exactly 2 operands, got ${operands.size}",
                "Unsupported add arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.add ${operands[0]}, ${operands[1]} : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    private fun convertMatmul(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Matmul operation requires exactly 2 operands, got ${operands.size}",
                "Unsupported matmul arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?xf32>"
        
        val resultValue = context.nextTempValue()
        
        // Emit dot_general with default contracting dimensions
        val operation1 = "$resultValue = stablehlo.dot_general ${operands[0]}, ${operands[1]}"
        val operation2 = "  contracting_dims = [[-1], [-2]] : $outputType"
        
        context.emitOperation(operation1)
        context.emitOperation(operation2)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation1, operation2)
        )
    }
    
    private fun convertRelu(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "ReLU operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported relu arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        val elementType = outputSpec?.let { context.getTypeMapper().mapDType(it.dtype) } 
            ?: "f32"
        
        val zeroConstValue = context.nextTempValue()
        val resultValue = context.nextTempValue()
        
        val operation1 = "$zeroConstValue = stablehlo.constant dense<0.0> : $outputType"
        val operation2 = "$resultValue = stablehlo.maximum ${operands[0]}, $zeroConstValue : $outputType"
        
        context.emitOperation(operation1)
        context.emitOperation(operation2)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation1, operation2)
        )
    }
}