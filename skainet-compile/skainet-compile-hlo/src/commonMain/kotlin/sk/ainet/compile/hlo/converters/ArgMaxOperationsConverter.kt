package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for `argMax` — the index of the maximum value along a dimension,
 * with ties resolved to the LOWEST index (numpy/greedy semantics). The reduced
 * dimension is removed from the output; the result is an `i32` index tensor.
 *
 * StableHLO has no argmax primitive, so — like softmax and the attention mask —
 * it is lowered by COMPOSING single-op primitives that this codebase already
 * emits (no variadic `stablehlo.reduce` with a custom reducer region, which the
 * MLIR validator here has no precedent for):
 *
 * ```
 *   maxV  = reduce(x)      applies stablehlo.maximum across [dim]   // per-position max value
 *   maxB  = broadcast_in_dim maxV -> input shape
 *   isMax = compare EQ, x, maxB                                     // i1 mask of maxima
 *   idx   = iota dim = dim                                          // i32 indices along dim
 *   cand  = select isMax, idx, <dimSize sentinel>                   // non-maxima -> out-of-range
 *   argIdx = reduce(cand)  applies stablehlo.minimum across [dim]   // lowest index of the max
 * ```
 *
 * The `compare EQ` is exact: `maxB` holds bit-identical copies of actual input
 * values, so the max element(s) compare equal; `minimum` then picks the lowest
 * index among ties and ignores non-maxima (masked to the `dimSize` sentinel,
 * which exceeds every valid `0..dimSize-1` index).
 */
public class ArgMaxOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf("argMax", "argmax")

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "argMax requires exactly 1 operand, got ${operands.size}",
                "Unsupported argMax arity for node ${node.id}"
            )
        }
        val inputSpec = node.inputs.firstOrNull()
            ?: return ConversionResult.Failure("argMax: missing input spec", "node ${node.id}")
        val outputSpec = node.outputs.firstOrNull()
            ?: return ConversionResult.Failure("argMax: missing output spec", "node ${node.id}")
        val inShape = inputSpec.shape
            ?: return ConversionResult.Failure("argMax: static input shape required", "node ${node.id}")

        val rank = inShape.size
        val dimParam = (node.operation.parameters["dim"] as? Int)
            ?: (node.operation.parameters["axis"] as? Int)
            ?: (rank - 1)
        val nd = if (dimParam < 0) rank + dimParam else dimParam
        if (nd !in 0 until rank) {
            return ConversionResult.Failure("argMax: dim $dimParam out of range for rank $rank", "node ${node.id}")
        }

        val tm = context.getTypeMapper()
        val valElem = tm.mapDType(inputSpec.dtype)     // e.g. "f32"
        val idxElem = tm.mapDType(outputSpec.dtype)    // "i32"
        fun typeOf(dims: List<Int>, elem: String): String =
            if (dims.isEmpty()) "tensor<$elem>" else "tensor<${dims.joinToString("x")}x$elem>"

        val fullValType = typeOf(inShape, valElem)                 // tensor<1x24x262153xf32>
        val fullIdxType = typeOf(inShape, idxElem)                 // tensor<1x24x262153xi32>
        val fullI1Type = typeOf(inShape, "i1")                     // tensor<1x24x262153xi1>
        val reducedDims = inShape.filterIndexed { i, _ -> i != nd }
        val reducedValType = typeOf(reducedDims, valElem)          // tensor<1x24xf32>
        val idxOutType = tm.mapTensorType(outputSpec)              // tensor<1x24xi32>
        val broadcastDims = (0 until rank).filter { it != nd }.joinToString(", ")
        val dimSize = inShape[nd]                                  // sentinel: > any valid index
        val negInf = tm.negInfBits(valElem)

        val ops = mutableListOf<String>()
        fun emit(op: String) { ops += op; context.emitOperation(op) }

        val maxInit = context.nextTempValue()
        emit("$maxInit = stablehlo.constant dense<$negInf> : tensor<$valElem>")
        val maxV = context.nextTempValue()
        emit(
            "$maxV = stablehlo.reduce(${operands[0]} init: $maxInit) " +
                "applies stablehlo.maximum across dimensions = [$nd] : " +
                "($fullValType, tensor<$valElem>) -> $reducedValType"
        )
        val maxB = context.nextTempValue()
        emit(
            "$maxB = stablehlo.broadcast_in_dim $maxV, dims = [$broadcastDims] : " +
                "($reducedValType) -> $fullValType"
        )
        val isMax = context.nextTempValue()
        emit("$isMax = stablehlo.compare EQ, ${operands[0]}, $maxB : ($fullValType, $fullValType) -> $fullI1Type")
        val idx = context.nextTempValue()
        emit("$idx = stablehlo.iota dim = $nd : $fullIdxType")
        val sentinel = context.nextTempValue()
        emit("$sentinel = stablehlo.constant dense<$dimSize> : $fullIdxType")
        val cand = context.nextTempValue()
        emit("$cand = stablehlo.select $isMax, $idx, $sentinel : $fullI1Type, $fullIdxType")
        val minInit = context.nextTempValue()
        emit("$minInit = stablehlo.constant dense<$dimSize> : tensor<$idxElem>")
        val argIdx = context.nextTempValue()
        emit(
            "$argIdx = stablehlo.reduce($cand init: $minInit) " +
                "applies stablehlo.minimum across dimensions = [$nd] : " +
                "($fullIdxType, tensor<$idxElem>) -> $idxOutType"
        )

        return ConversionResult.Success(outputValueName = argIdx, emittedOperations = ops)
    }
}
