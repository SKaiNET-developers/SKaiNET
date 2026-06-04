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
 * Causal masking: when the `causal` attribute is set, an additive -inf mask
 * (built from iota row/col indices + compare + select) is added to the scaled
 * scores before softmax so each query only attends to keys at or before it.
 * An explicit `mask` operand is not yet consumed (TODO: add operands[3]).
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

        val causal = (node.operation.parameters["causal"] as? Boolean) ?: false
        val qAxis = rank - 2 // query position in scores [.., Sq, Sk]
        val scoresI32Type = "tensor<${scoresShape.joinToString("x")}xi32>"
        val scoresI1Type = "tensor<${scoresShape.joinToString("x")}xi1>"

        val scores = context.nextTempValue()
        val scaleC = context.nextTempValue()
        val scaled = context.nextTempValue()
        val maxInit = context.nextTempValue(); val maxV = context.nextTempValue(); val maxB = context.nextTempValue()
        val shifted = context.nextTempValue(); val expV = context.nextTempValue()
        val sumInit = context.nextTempValue(); val sumV = context.nextTempValue(); val sumB = context.nextTempValue()
        val attn = context.nextTempValue()
        val out = context.nextTempValue()

        val ops = mutableListOf(
            "$scores = stablehlo.dot_general ${operands[0]}, ${operands[1]}, ${batchClause}contracting_dims = [$contractQK] x [$contractQK] : ($qType, $kType) -> $scoresType",
            "$scaleC = stablehlo.constant dense<$scaleVal> : $scoresType",
            "$scaled = stablehlo.multiply $scores, $scaleC : $scoresType",
        )

        // Explicit additive mask (operands[3]) — e.g. a sliding-window+causal
        // mask the caller built and passed with causal=false. It already
        // encodes causality/window, so it takes priority over the built-in
        // iota causal path. Broadcast (trailing-aligned) to the scores shape
        // and add. Without this the masked layers run UNMASKED (attend to
        // future tokens) — correct only at position 0.
        var softmaxIn = scaled
        val maskOperand = operands.getOrNull(3)
        if (maskOperand != null) {
            val maskShape = node.inputs.getOrNull(3)?.shape ?: scoresShape
            val maskType = context.getValueType(maskOperand) ?: typeOf(maskShape)
            val maskBc = if (maskShape == scoresShape) {
                maskOperand
            } else {
                val mb = context.nextTempValue()
                val offset = scoresShape.size - maskShape.size
                val dims = maskShape.indices.joinToString(", ") { (it + offset).toString() }
                ops += "$mb = stablehlo.broadcast_in_dim $maskOperand, dims = [$dims] : ($maskType) -> $scoresType"
                mb
            }
            val masked = context.nextTempValue()
            ops += "$masked = stablehlo.add $scaled, $maskBc : $scoresType"
            softmaxIn = masked
        } else if (causal) {
            val iotaQ = context.nextTempValue(); val iotaK = context.nextTempValue()
            val keep = context.nextTempValue(); val zeros = context.nextTempValue()
            val ninf = context.nextTempValue(); val maskAdd = context.nextTempValue()
            val masked = context.nextTempValue()
            ops += "$iotaQ = stablehlo.iota dim = $qAxis : $scoresI32Type"
            ops += "$iotaK = stablehlo.iota dim = $sdAxis : $scoresI32Type"
            ops += "$keep = stablehlo.compare GE, $iotaQ, $iotaK : ($scoresI32Type, $scoresI32Type) -> $scoresI1Type"
            ops += "$zeros = stablehlo.constant dense<0.0> : $scoresType"
            ops += "$ninf = stablehlo.constant dense<0xFF800000> : $scoresType"
            ops += "$maskAdd = stablehlo.select $keep, $zeros, $ninf : $scoresI1Type, $scoresType"
            ops += "$masked = stablehlo.add $scaled, $maskAdd : $scoresType"
            softmaxIn = masked
        }

        // softmax(softmaxIn) over the key-length axis
        ops += "$maxInit = stablehlo.constant dense<0xFF800000> : tensor<$elem>"
        ops += "$maxV = stablehlo.reduce($softmaxIn init: $maxInit) applies stablehlo.maximum across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType"
        ops += "$maxB = stablehlo.broadcast_in_dim $maxV, dims = [$bcastDims] : ($reducedType) -> $scoresType"
        ops += "$shifted = stablehlo.subtract $softmaxIn, $maxB : $scoresType"
        ops += "$expV = stablehlo.exponential $shifted : $scoresType"
        ops += "$sumInit = stablehlo.constant dense<0.0> : tensor<$elem>"
        ops += "$sumV = stablehlo.reduce($expV init: $sumInit) applies stablehlo.add across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType"
        ops += "$sumB = stablehlo.broadcast_in_dim $sumV, dims = [$bcastDims] : ($reducedType) -> $scoresType"
        ops += "$attn = stablehlo.divide $expV, $sumB : $scoresType"
        ops += "$out = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outputType"
        ops.forEach { context.emitOperation(it) }
        return ConversionResult.Success(outputValueName = out, emittedOperations = ops)
    }
}
