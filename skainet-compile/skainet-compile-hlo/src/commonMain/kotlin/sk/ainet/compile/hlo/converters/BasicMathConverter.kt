package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.compile.hlo.TypeMapper
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Converter for basic mathematical operations (add, subtract, multiply, divide).
 *
 * This converter handles the basic arithmetic operations by mapping them
 * to their corresponding StableHLO operations. It supports:
 * - Element-wise operations with broadcasting
 * - Mixed-type arithmetic with automatic type promotion
 * - Rank-differing operands (e.g. bias `tensor<C>` added to conv output
 *   `tensor<N,C,L>`) — emits `stablehlo.broadcast_in_dim` before the op so
 *   iree-compile accepts the MLIR.
 */
public class BasicMathConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        "add", "subtract", "multiply", "divide",
        "sub", "mul", "div", // Common aliases
        "pow"
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

        val typeMapper = context.getTypeMapper()
        val stableHloOp = getStableHloOperation(node.operation.name)
            ?: return ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by BasicMathConverter"
            )

        // Prefer the node's recorded output spec (carries the broadcasted shape); fall back
        // to inferring from the two inputs.
        val targetSpec = node.outputs.firstOrNull()
            ?: typeMapper.inferBroadcastType(leftSpec, rightSpec)
        val resultType = typeMapper.mapTensorType(targetSpec)

        val adaptedLeft = adaptOperand(operands[0], leftSpec, targetSpec, typeMapper, context)
        val adaptedRight = adaptOperand(operands[1], rightSpec, targetSpec, typeMapper, context)

        val resultValue = context.nextTempValue()
        val operation = "$resultValue = $stableHloOp $adaptedLeft, $adaptedRight : $resultType"
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
            "pow" -> "stablehlo.power"
            else -> null
        }
    }

    /**
     * Make `operand` match `targetSpec` by emitting `stablehlo.convert` (dtype promotion)
     * and/or `stablehlo.broadcast_in_dim` (shape broadcast) as needed.
     */
    private fun adaptOperand(
        operand: String,
        sourceSpec: TensorSpec,
        targetSpec: TensorSpec,
        typeMapper: TypeMapper,
        context: ConversionContext
    ): String {
        var current = operand
        var currentSpec = sourceSpec

        // Step 1: dtype promotion
        if (typeMapper.mapDType(currentSpec.dtype) != typeMapper.mapDType(targetSpec.dtype)) {
            val converted = context.nextTempValue()
            val fromType = typeMapper.mapTensorType(currentSpec)
            val promotedSpec = TensorSpec(
                name = "${currentSpec.name}_conv",
                shape = currentSpec.shape,
                dtype = targetSpec.dtype
            )
            val toType = typeMapper.mapTensorType(promotedSpec)
            context.emitOperation("$converted = stablehlo.convert $current : $fromType to $toType")
            current = converted
            currentSpec = promotedSpec
        }

        // Step 2: shape broadcast (only if shapes differ and both are known)
        if (currentSpec.shape != null && targetSpec.shape != null &&
            currentSpec.shape != targetSpec.shape
        ) {
            val broadcast = context.nextTempValue()
            val dims = computeBroadcastDims(currentSpec.shape, targetSpec.shape)
            val fromType = typeMapper.mapTensorType(currentSpec)
            val toType = typeMapper.mapTensorType(targetSpec)
            context.emitOperation(
                "$broadcast = stablehlo.broadcast_in_dim $current, dims = [$dims] : " +
                    "($fromType) -> $toType"
            )
            current = broadcast
        }

        return current
    }

    /**
     * NumPy-style right-aligned broadcast: map each source axis `i` to target axis
     * `targetRank - sourceRank + i`. Scalars (rank 0) map to an empty dim list.
     */
    private fun computeBroadcastDims(sourceShape: List<Int>?, targetShape: List<Int>?): String {
        if (sourceShape == null || targetShape == null) return ""
        val sourceDims = sourceShape.size
        val targetDims = targetShape.size
        if (sourceDims == 0) return ""
        val offset = targetDims - sourceDims
        return (offset until targetDims).joinToString(", ")
    }
}
