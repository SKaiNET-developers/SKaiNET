package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode
import kotlin.math.sqrt

/**
 * Converts `scaledDotProductAttention` to the standard StableHLO attention
 * subgraph:
 *
 *     scores = Q · Kᵀ                      (dot_general, contract head_dim)
 *     scaled = scores * scale              (scale = arg, or 1/sqrt(head_dim))
 *     attn   = softmax(scaled, axis = -1)  (max/sub/exp/sum/div, numerically stable)
 *     out    = attn · V                    (dot_general, contract key length)
 *
 * Q/K/V are batched `[.., S, D]`; batching dims are every leading dim except the
 * last two. The softmax decomposition mirrors ActivationOperationsConverter.
 *
 * SDPA is a core `TensorOps` op (KSP-generated), so its converter lives here in
 * core alongside dot_general/softmax — the transformer modules just decompose to it.
 *
 * v1 limitation: the optional attention `mask` / `causal` flag is not yet
 * emitted (structurally correct, numerically unmasked). TODO: emit a causal mask
 * (iota + compare + select, additive -inf) before the softmax.
 */
public class AttentionOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        "scaledDotProductAttention", "scaleddotproductattention", "sdpa"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 3) {
            return ConversionResult.Failure(
                "scaledDotProductAttention requires q, k, v (>=3 operands), got ${operands.size}",
                "Unsupported SDPA arity for node ${node.id}"
            )
        }

        val qShape = node.inputs.getOrNull(0)?.shape
            ?: return ConversionResult.Failure("SDPA requires a known query shape", "Missing query shape for ${node.id}")
        val kShape = node.inputs.getOrNull(1)?.shape ?: qShape
        val vShape = node.inputs.getOrNull(2)?.shape ?: qShape
        val rank = qShape.size
        if (rank < 2) {
            return ConversionResult.Failure("SDPA query must have rank >= 2", "Bad query rank for ${node.id}")
        }

        val outSpec = node.outputs.firstOrNull()
        val mapper = context.getTypeMapper()
        val elem = outSpec?.let { mapper.mapDType(it.dtype) } ?: "f32"
        fun typeOf(shape: List<Int>): String = "tensor<${shape.joinToString("x")}x$elem>"

        val qType = context.getValueType(operands[0]) ?: typeOf(qShape)
        val kType = context.getValueType(operands[1]) ?: typeOf(kShape)
        val vType = context.getValueType(operands[2]) ?: typeOf(vShape)

        val headDim = qShape[rank - 1]
        val keyLen = kShape[rank - 2]
        val scoresShape = qShape.dropLast(1) + keyLen   // [.., Sq, Sk]
        val scoresType = typeOf(scoresShape)
        val outputType = outSpec?.let { mapper.mapTensorType(it) } ?: typeOf(qShape.dropLast(1) + headDim)

        val scaleParam = (node.operation.parameters["scale"] as? Number)?.toFloat() ?: 0f
        val scaleVal = if (scaleParam != 0f) scaleParam else (1.0f / sqrt(headDim.toFloat()))

        val hasBatch = rank > 2
        val batchList = (0 until rank - 2).joinToString(", ")
        val batchClause = if (hasBatch) "batching_dims = [$batchList] x [$batchList], " else ""
        val contractQK = rank - 1                 // contract head_dim of Q and K
        val sdAxis = scoresShape.size - 1         // softmax over key length
        val reducedShape = scoresShape.dropLast(1)
        val reducedType = if (reducedShape.isEmpty()) "tensor<$elem>" else "tensor<${reducedShape.joinToString("x")}x$elem>"
        val bcastDims = (scoresShape.indices).filter { it != sdAxis }.joinToString(", ")
        val contractAttn = scoresShape.size - 1   // attn key-length axis
        val contractV = rank - 2                  // V key-length axis

        val scores = context.nextTempValue()
        val scaleC = context.nextTempValue()
        val scaled = context.nextTempValue()
        val maxInit = context.nextTempValue(); val maxV = context.nextTempValue(); val maxB = context.nextTempValue()
        val shifted = context.nextTempValue(); val expV = context.nextTempValue()
        val sumInit = context.nextTempValue(); val sumV = context.nextTempValue(); val sumB = context.nextTempValue()
        val attn = context.nextTempValue()
        val out = context.nextTempValue()

        val ops = listOf(
            "$scores = stablehlo.dot_general ${operands[0]}, ${operands[1]}, ${batchClause}contracting_dims = [$contractQK] x [$contractQK] : ($qType, $kType) -> $scoresType",
            "$scaleC = stablehlo.constant dense<$scaleVal> : $scoresType",
            "$scaled = stablehlo.multiply $scores, $scaleC : $scoresType",
            // softmax(scaled) over the key-length axis
            "$maxInit = stablehlo.constant dense<0xFF800000> : tensor<$elem>",
            "$maxV = stablehlo.reduce($scaled init: $maxInit) applies stablehlo.maximum across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType",
            "$maxB = stablehlo.broadcast_in_dim $maxV, dims = [$bcastDims] : ($reducedType) -> $scoresType",
            "$shifted = stablehlo.subtract $scaled, $maxB : $scoresType",
            "$expV = stablehlo.exponential $shifted : $scoresType",
            "$sumInit = stablehlo.constant dense<0.0> : tensor<$elem>",
            "$sumV = stablehlo.reduce($expV init: $sumInit) applies stablehlo.add across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType",
            "$sumB = stablehlo.broadcast_in_dim $sumV, dims = [$bcastDims] : ($reducedType) -> $scoresType",
            "$attn = stablehlo.divide $expV, $sumB : $scoresType",
            "$out = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outputType",
        )
        ops.forEach { context.emitOperation(it) }
        return ConversionResult.Success(outputValueName = out, emittedOperations = ops)
    }
}
