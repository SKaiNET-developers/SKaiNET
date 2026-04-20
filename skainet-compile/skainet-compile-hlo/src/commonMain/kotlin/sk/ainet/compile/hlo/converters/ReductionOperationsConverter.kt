package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Converter for reduction operations (sum, mean, variance).
 *
 * Emits real `stablehlo.reduce` ops (pretty-printed short form), not
 * `stablehlo.custom_call @reduce_*`. The custom_call form is a non-standard
 * target that `iree-compile` rejects; this was the first compile error on
 * the Whisper encoder MLIR. `mean` and `variance` are decomposed into reduce
 * + arithmetic since StableHLO only has a single `stablehlo.reduce` primitive
 * with a body.
 *
 * Reduced shape semantics match `VoidTensorOps.calculateReductionShape`:
 * the reduced dimension is removed from the output (no keepdim). If `dim`
 * is null the result is a scalar (tensor<f32>).
 */
public class ReductionOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf("sum", "mean", "variance")

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
        val inputSpec = node.inputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Sum: missing input spec",
                "Unsupported sum for node ${node.id}"
            )
        val outputSpec = node.outputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Sum: missing output spec",
                "Unsupported sum for node ${node.id}"
            )

        val dims = reductionDims(node, inputSpec)
        val ops = mutableListOf<String>()
        val result = emitReduce(
            input = operands[0],
            inputSpec = inputSpec,
            outputSpec = outputSpec,
            combinator = "stablehlo.add",
            initLiteral = "0.0",
            dims = dims,
            context = context,
            outOps = ops
        )
        return ConversionResult.Success(outputValueName = result, emittedOperations = ops)
    }

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
        val inputSpec = node.inputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Mean: missing input spec",
                "Unsupported mean for node ${node.id}"
            )
        val outputSpec = node.outputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Mean: missing output spec",
                "Unsupported mean for node ${node.id}"
            )

        val dims = reductionDims(node, inputSpec)
        val count = reductionCount(inputSpec, dims)
        val outputType = context.getTypeMapper().mapTensorType(outputSpec)

        val ops = mutableListOf<String>()
        val sumValue = emitReduce(
            input = operands[0],
            inputSpec = inputSpec,
            outputSpec = outputSpec,
            combinator = "stablehlo.add",
            initLiteral = "0.0",
            dims = dims,
            context = context,
            outOps = ops
        )

        val countValue = context.nextTempValue()
        val countOp = "$countValue = stablehlo.constant dense<$count> : $outputType"
        ops += countOp
        context.emitOperation(countOp)

        val resultValue = context.nextTempValue()
        val divOp = "$resultValue = stablehlo.divide $sumValue, $countValue : $outputType"
        ops += divOp
        context.emitOperation(divOp)

        return ConversionResult.Success(outputValueName = resultValue, emittedOperations = ops)
    }

    /**
     * Variance via E[X^2] − E[X]^2. Uses two `stablehlo.reduce` passes rather
     * than the centered-squared form because the latter would require a
     * `broadcast_in_dim` back to the input rank.
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
        val inputSpec = node.inputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Variance: missing input spec",
                "Unsupported variance for node ${node.id}"
            )
        val outputSpec = node.outputs.firstOrNull()
            ?: return ConversionResult.Failure(
                "Variance: missing output spec",
                "Unsupported variance for node ${node.id}"
            )

        val typeMapper = context.getTypeMapper()
        val dims = reductionDims(node, inputSpec)
        val count = reductionCount(inputSpec, dims)
        val inputType = typeMapper.mapTensorType(inputSpec)
        val outputType = typeMapper.mapTensorType(outputSpec)

        val ops = mutableListOf<String>()

        // sq = x * x
        val sqValue = context.nextTempValue()
        val sqOp = "$sqValue = stablehlo.multiply ${operands[0]}, ${operands[0]} : $inputType"
        ops += sqOp
        context.emitOperation(sqOp)

        // E[X]  = reduce_sum(x)  / N
        val sumX = emitReduce(
            input = operands[0], inputSpec = inputSpec, outputSpec = outputSpec,
            combinator = "stablehlo.add", initLiteral = "0.0",
            dims = dims, context = context, outOps = ops
        )
        // E[X^2] = reduce_sum(x*x) / N
        val sumSq = emitReduce(
            input = sqValue, inputSpec = inputSpec, outputSpec = outputSpec,
            combinator = "stablehlo.add", initLiteral = "0.0",
            dims = dims, context = context, outOps = ops
        )

        val countValue = context.nextTempValue()
        val countOp = "$countValue = stablehlo.constant dense<$count> : $outputType"
        ops += countOp
        context.emitOperation(countOp)

        val meanX = context.nextTempValue()
        val meanXOp = "$meanX = stablehlo.divide $sumX, $countValue : $outputType"
        ops += meanXOp
        context.emitOperation(meanXOp)

        val meanSq = context.nextTempValue()
        val meanSqOp = "$meanSq = stablehlo.divide $sumSq, $countValue : $outputType"
        ops += meanSqOp
        context.emitOperation(meanSqOp)

        val meanSquared = context.nextTempValue()
        val meanSquaredOp = "$meanSquared = stablehlo.multiply $meanX, $meanX : $outputType"
        ops += meanSquaredOp
        context.emitOperation(meanSquaredOp)

        val resultValue = context.nextTempValue()
        val subOp = "$resultValue = stablehlo.subtract $meanSq, $meanSquared : $outputType"
        ops += subOp
        context.emitOperation(subOp)

        return ConversionResult.Success(outputValueName = resultValue, emittedOperations = ops)
    }

    /**
     * Derive the list of reduction dimensions from op parameters. If neither
     * `dim` nor `axis` is present, reduce all dimensions (full reduction).
     */
    private fun reductionDims(node: GraphNode, inputSpec: TensorSpec): List<Int> {
        val params = node.operation.parameters
        val dim = params["dim"] as? Int ?: params["axis"] as? Int
        if (dim != null) {
            val rank = inputSpec.shape?.size ?: 0
            val resolved = if (dim < 0) rank + dim else dim
            return listOf(resolved)
        }
        val rank = inputSpec.shape?.size ?: 0
        return (0 until rank).toList()
    }

    /** Number of elements collapsed into one output element. */
    private fun reductionCount(inputSpec: TensorSpec, dims: List<Int>): Double {
        val shape = inputSpec.shape ?: return 1.0
        var n = 1L
        for (d in dims) {
            if (d in shape.indices) n *= shape[d].toLong()
        }
        return n.toDouble()
    }

    /**
     * Emit `stablehlo.constant` for the reduction init value, then
     * `stablehlo.reduce(... init: ...) applies COMBINATOR across dimensions = [...]`.
     * Returns the SSA value of the reduce result and appends every emitted line
     * to `outOps`.
     */
    private fun emitReduce(
        input: String,
        inputSpec: TensorSpec,
        outputSpec: TensorSpec,
        combinator: String,
        initLiteral: String,
        dims: List<Int>,
        context: ConversionContext,
        outOps: MutableList<String>
    ): String {
        val typeMapper = context.getTypeMapper()
        val elementType = typeMapper.mapDType(outputSpec.dtype)
        val inputType = typeMapper.mapTensorType(inputSpec)
        val outputType = typeMapper.mapTensorType(outputSpec)

        val initValue = context.nextTempValue()
        val initOp = "$initValue = stablehlo.constant dense<$initLiteral> : tensor<$elementType>"
        outOps += initOp
        context.emitOperation(initOp)

        val resultValue = context.nextTempValue()
        val dimsStr = dims.joinToString(", ")
        val reduceOp = "$resultValue = stablehlo.reduce($input init: $initValue) " +
            "applies $combinator across dimensions = [$dimsStr] : " +
            "($inputType, tensor<$elementType>) -> $outputType"
        outOps += reduceOp
        context.emitOperation(reduceOp)

        return resultValue
    }
}
