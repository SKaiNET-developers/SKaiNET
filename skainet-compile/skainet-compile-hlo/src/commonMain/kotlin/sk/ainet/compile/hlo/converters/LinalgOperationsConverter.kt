package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for linear algebra operations.
 * 
 * This converter implements support for matrix operations including matrix multiplication
 * and transpose operations as specified in Requirements 2.2:
 * - Matrix multiplication (matmul) using stablehlo.dot_general
 * - Transpose operations with arbitrary dimension permutations
 * - Batch matrix operations support
 * - Proper dot_general configuration for contracting dimensions
 * 
 * The converter handles:
 * - 2D matrix multiplication (standard matmul)
 * - Batch matrix multiplication (3D+ tensors)
 * - Transpose with configurable dimension permutations
 * - Proper type inference and shape handling
 */
public class LinalgOperationsConverter : StableHloOperationConverter {
    
    override val supportedOperations: Set<String> = setOf(
        "matmul", "transpose",
        // Common aliases
        "dot", "mm", "bmm", "batch_matmul"
    )
    
    override fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "matmul", "dot", "mm" -> convertMatmul(node, operands, context)
            "bmm", "batch_matmul" -> convertBatchMatmul(node, operands, context)
            "transpose" -> convertTranspose(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by LinalgOperationsConverter"
            )
        }
    }
    
    /**
     * Convert standard matrix multiplication to stablehlo.dot_general.
     * 
     * For 2D matrices A (M x K) and B (K x N), produces C (M x N) where:
     * - Contracting dimensions: last dim of A ([1]) with second-to-last dim of B ([0])
     * - This follows the standard matrix multiplication convention
     */
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
        
        // Standard matmul: contract last dimension of left operand with 
        // second-to-last dimension of right operand
        // For 2D: A[M,K] x B[K,N] -> C[M,N]
        // contracting_dims = [[1], [0]] means:
        //   - dimension 1 (K) of left operand
        //   - dimension 0 (K) of right operand
        val operation = "$resultValue = stablehlo.dot_general ${operands[0]}, ${operands[1]}, contracting_dims = [[1], [0]] : $outputType"
        
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert batch matrix multiplication to stablehlo.dot_general.
     * 
     * For 3D+ tensors with batch dimensions, performs batched matrix multiplication.
     * For example, A (B x M x K) and B (B x K x N) produces C (B x M x N) where:
     * - Batch dimensions: [0] (the batch dimension is preserved)
     * - Contracting dimensions: last dim of A with second-to-last dim of B
     */
    private fun convertBatchMatmul(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Batch matmul operation requires exactly 2 operands, got ${operands.size}",
                "Unsupported batch matmul arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?x?xf32>"
        
        // Determine batch dimensions from the operation parameters or infer from shapes
        val batchDims = node.operation.parameters["batch_dims"] as? List<*>
        val batchDimsStr = if (batchDims != null && batchDims.isNotEmpty()) {
            val dims = batchDims.joinToString(", ")
            "[[${dims}], [${dims}]]"
        } else {
            // Default: assume first dimension is batch
            "[[0], [0]]"
        }
        
        val resultValue = context.nextTempValue()
        
        // Batch matmul: preserve batch dimensions and contract matrix dimensions
        // For 3D: A[B,M,K] x B[B,K,N] -> C[B,M,N]
        // batch_dims = [[0], [0]] means batch dimension 0 is preserved
        // contracting_dims = [[2], [1]] means:
        //   - dimension 2 (K) of left operand
        //   - dimension 1 (K) of right operand
        val operation = "$resultValue = stablehlo.dot_general ${operands[0]}, ${operands[1]}, contracting_dims = [[2], [1]], batch_dims = $batchDimsStr : $outputType"
        
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    /**
     * Convert transpose operation to stablehlo.transpose.
     * 
     * Handles arbitrary dimension permutations. The permutation can be specified
     * in the operation parameters, or defaults to reversing all dimensions.
     * 
     * For example:
     * - 2D transpose: [0, 1] -> [1, 0]
     * - 3D transpose: [0, 1, 2] -> [2, 1, 0] (default)
     * - Custom: [0, 1, 2] -> [0, 2, 1] (transpose last two dims)
     */
    private fun convertTranspose(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Transpose operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported transpose arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?x?xf32>"
        
        // Get permutation from operation parameters
        val permutation = node.operation.parameters["permutation"] as? List<*>
            ?: node.operation.parameters["perm"] as? List<*>
            ?: node.operation.parameters["axes"] as? List<*>
        
        val permutationStr = if (permutation != null && permutation.isNotEmpty()) {
            // Use provided permutation
            permutation.joinToString(", ")
        } else {
            // Default: reverse all dimensions
            // For 2D: [1, 0], for 3D: [2, 1, 0], etc.
            val inputSpec = context.getInputNodes(node).firstOrNull()?.outputs?.firstOrNull()
            val rank = inputSpec?.shape?.size ?: 2
            (rank - 1 downTo 0).joinToString(", ")
        }
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.transpose ${operands[0]}, dims = [$permutationStr] : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
}
