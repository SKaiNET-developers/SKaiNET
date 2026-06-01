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
        "dot", "mm", "bmm", "batch_matmul",
        // permute is an arbitrary-axis transpose; convertTranspose already
        // reads the `axes` parameter, so route it through the same lowering.
        "permute"
    )
    
    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            // All matmul variants share the same lowering: the batched form
            // degenerates to the compact rank-2 form when rank == 2 (no
            // batching_dims emitted). Whisper's attention uses rank-4 matmul
            // where `[1] x [0]` is wrong — contract last/second-to-last.
            "matmul", "dot", "mm", "bmm", "batch_matmul" ->
                convertBatchMatmul(node, operands, context)
            "transpose", "permute" -> convertTranspose(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by LinalgOperationsConverter"
            )
        }
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
        val lhsSpec = node.inputs.getOrNull(0)
        val rhsSpec = node.inputs.getOrNull(1)
        val lhsType = lhsSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?x?x?xf32>"
        val rhsType = rhsSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?x?x?xf32>"

        // Infer batching rank: for A[..., M, K] x B[..., K, N], batching dims
        // are all leading dims except the last two. Falls back to rank 3 if
        // shape is unknown (matches prior hard-coded behavior).
        val rank = lhsSpec?.shape?.size ?: rhsSpec?.shape?.size ?: 3
        val batchCount = (rank - 2).coerceAtLeast(0)
        val explicitBatch = node.operation.parameters["batch_dims"] as? List<*>
        val batchDimsList = if (explicitBatch != null && explicitBatch.isNotEmpty()) {
            explicitBatch.map { it.toString() }
        } else {
            (0 until batchCount).map { it.toString() }
        }
        val contractingLhs = rank - 1
        val contractingRhs = (rank - 2).coerceAtLeast(0)

        val resultValue = context.nextTempValue()

        // Batch matmul: preserve batch dimensions and contract matrix dimensions.
        // For A[..., M, K] x B[..., K, N]: batching_dims cover leading dims,
        // contracting_dims = [rank-1] x [rank-2].
        val batchClause = if (batchDimsList.isNotEmpty()) {
            val b = batchDimsList.joinToString(", ")
            "batching_dims = [$b] x [$b], "
        } else {
            ""
        }
        val operation = "$resultValue = stablehlo.dot_general ${operands[0]}, ${operands[1]}, ${batchClause}contracting_dims = [$contractingLhs] x [$contractingRhs] : ($lhsType, $rhsType) -> $outputType"

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
        val inputSpec = node.inputs.firstOrNull()
        val inputType = inputSpec?.let { context.getTypeMapper().mapTensorType(it) }
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
            val rank = inputSpec?.shape?.size ?: 2
            (rank - 1 downTo 0).joinToString(", ")
        }

        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.transpose ${operands[0]}, dims = [$permutationStr] : ($inputType) -> $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
}
