package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for reduction operations.
 *
 * This converter handles reduction operations using StableHLO primitives:
 * - sum: using stablehlo.custom_call @reduce_sum with dimension attributes
 * - mean: using reduce_sum + divide by element count
 * - variance: using stablehlo.custom_call @reduce_variance
 *
 * Uses custom_call for reduction operations that require region bodies,
 * consistent with the approach used by NeuralNetOperationsConverter for layerNorm.
 */
public class ReductionOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        "sum", "mean", "variance"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            "sum" -> convertSum(node, operands, context)
            "mean" -> convertMean(node, operands, context)
            "variance" -> convertVariance(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by ReductionOperationsConverter"
            )
        }
    }

    /**
     * Convert sum reduction using stablehlo.custom_call @reduce_sum.
     */
    private fun convertSum(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Sum operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported sum arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val params = node.operation.parameters
        val dim = params["dim"] as? Int ?: params["axis"] as? Int
        val keepdim = params["keepdim"] as? Boolean ?: false

        val resultValue = context.nextTempValue()

        val dimAttr = if (dim != null) {
            "dimensions = [$dim]"
        } else {
            "dimensions = []"
        }
        val keepdimAttr = "keepdim = $keepdim"

        val operation = "$resultValue = stablehlo.custom_call @reduce_sum(${operands[0]}) " +
                "{$dimAttr, $keepdimAttr} : $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }

    /**
     * Convert mean reduction using reduce_sum + divide by element count.
     */
    private fun convertMean(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Mean operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported mean arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val params = node.operation.parameters
        val dim = params["dim"] as? Int ?: params["axis"] as? Int
        val keepdim = params["keepdim"] as? Boolean ?: false

        val dimAttr = if (dim != null) {
            "dimensions = [$dim]"
        } else {
            "dimensions = []"
        }
        val keepdimAttr = "keepdim = $keepdim"

        // Step 1: compute the sum
        val sumValue = context.nextTempValue()
        val sumOp = "$sumValue = stablehlo.custom_call @reduce_sum(${operands[0]}) " +
                "{$dimAttr, $keepdimAttr} : $outputType"
        context.emitOperation(sumOp)

        // Step 2: compute element count along reduction dimension
        val inputShape = node.inputs.firstOrNull()?.shape
        val count = if (dim != null && inputShape != null && dim < inputShape.size) {
            inputShape[dim].toDouble()
        } else {
            // reduce-all: product of all dimensions
            inputShape?.fold(1) { acc, d -> acc * d }?.toDouble() ?: 1.0
        }

        val countValue = context.nextTempValue()
        val countOp = "$countValue = stablehlo.constant dense<$count> : $outputType"
        context.emitOperation(countOp)

        // Step 3: divide sum by count
        val resultValue = context.nextTempValue()
        val divOp = "$resultValue = stablehlo.divide $sumValue, $countValue : $outputType"
        context.emitOperation(divOp)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(sumOp, countOp, divOp)
        )
    }

    /**
     * Convert variance reduction using stablehlo.custom_call @reduce_variance.
     */
    private fun convertVariance(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Variance operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported variance arity for node ${node.id}"
            )
        }

        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) }
            ?: "tensor<?xf32>"

        val params = node.operation.parameters
        val dim = params["dim"] as? Int ?: params["axis"] as? Int
        val keepdim = params["keepdim"] as? Boolean ?: false

        val dimAttr = if (dim != null) {
            "dimensions = [$dim]"
        } else {
            "dimensions = []"
        }
        val keepdimAttr = "keepdim = $keepdim"

        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.custom_call @reduce_variance(${operands[0]}) " +
                "{$dimAttr, $keepdimAttr} : $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
}
