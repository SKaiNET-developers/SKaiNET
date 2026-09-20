package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.hasDynamic
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
 * An explicit `mask` operand (operands[3]) takes priority over the iota path.
 *
 * Grouped-query attention (SKEEP-005 phase 2, "structure at compile time"): when K/V
 * carry fewer heads than Q (`[b, nKV, Sk, hd]` vs `[b, H, Sq, hd]`, `nKV | H`), Q is
 * reshaped to `[b, nKV, nRep, Sq, hd]` and both `dot_general`s batch over `[b, nKV]`
 * with `nRep` a free axis — K/V are never broadcast or concatenated. The group
 * structure is exactly what the compiler backend tiles over; the core count is not
 * written anywhere (it is a run-time property of the device).
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
        // Render a shape's dims, mapping a dynamic extent (DYNAMIC_DIM = -1) to `?`.
        fun dims(shape: List<Int>): String = shape.joinToString("x") { Dim.render(it) }
        fun typeOf(shape: List<Int>): String = "tensor<${dims(shape)}x$elem>"

        val qType = context.getValueType(operands[0]) ?: typeOf(qShape)
        val kType = context.getValueType(operands[1]) ?: typeOf(kShape)
        val vType = context.getValueType(operands[2]) ?: typeOf(vShape)

        val headDim = qShape[rank - 1]
        val keyLen = kShape[rank - 2]

        // Grouped-query attention: K/V heads divide Q heads → work on Q as [b, nKV, nRep, Sq, hd].
        val gqa = rank == 4 && kShape.size == 4 && kShape[1] != qShape[1]
        if (gqa) {
            val nKV = kShape[1]; val nH = qShape[1]
            if (!Dim.isStatic(nKV) || !Dim.isStatic(nH) || nKV <= 0 || nH % nKV != 0) {
                return ConversionResult.Failure(
                    "SDPA grouped-query attention needs static head counts with K/V heads dividing Q heads, got Q=$nH K/V=$nKV",
                    "Unsupported GQA head counts for ${node.id}",
                )
            }
            if (qShape.hasDynamic()) {
                return ConversionResult.Failure(
                    "SDPA grouped-query attention needs a static query shape (reshape to [b, nKV, nRep, Sq, hd]), got $qShape",
                    "Dynamic query shape under GQA for ${node.id}",
                )
            }
        }
        val qWork: List<Int> = if (gqa) listOf(qShape[0], kShape[1], qShape[1] / kShape[1], qShape[2], headDim) else qShape
        val qWorkType = if (gqa) typeOf(qWork) else qType
        val scoresShape = qWork.dropLast(1) + keyLen   // [.., Sq, Sk]  (GQA: [b, nKV, nRep, Sq, Sk])
        val scoresType = typeOf(scoresShape)
        val outputType = outSpec?.let { mapper.mapTensorType(it) } ?: typeOf(qShape.dropLast(1) + headDim)
        val outWorkType = if (gqa) typeOf(qWork.dropLast(1) + headDim) else outputType

        val scaleParam = (node.operation.parameters["scale"] as? Number)?.toFloat() ?: 0f
        val scaleVal = if (scaleParam != 0f) scaleParam else (1.0f / sqrt(headDim.toFloat()))

        val hasBatch = rank > 2
        val batchList = (0 until rank - 2).joinToString(", ")   // K/V batching dims; == Q's ([b, nKV]) under GQA too
        val batchClause = if (hasBatch) "batching_dims = [$batchList] x [$batchList], " else ""
        val contractQ = qWork.size - 1            // contract head_dim of Q …
        val contractK = rank - 1                  // … and K
        val sdAxis = scoresShape.size - 1         // softmax over key length
        val reducedShape = scoresShape.dropLast(1)
        val reducedType = if (reducedShape.isEmpty()) "tensor<$elem>" else typeOf(reducedShape)
        val bcastDims = (scoresShape.indices).filter { it != sdAxis }.joinToString(", ")
        val contractAttn = scoresShape.size - 1   // attn key-length axis
        val contractV = rank - 2                  // V key-length axis

        val causal = (node.operation.parameters["causal"] as? Boolean) ?: false
        val qAxis = scoresShape.size - 2 // query position in scores [.., Sq, Sk]
        val scoresI32Type = "tensor<${dims(scoresShape)}xi32>"
        val scoresI1Type = "tensor<${dims(scoresShape)}xi1>"

        val scaleC = context.nextTempValue()
        val scaleB = context.nextTempValue()
        val qScaled = context.nextTempValue()
        val scores = context.nextTempValue()
        val maxInit = context.nextTempValue(); val maxV = context.nextTempValue(); val maxB = context.nextTempValue()
        val shifted = context.nextTempValue(); val expV = context.nextTempValue()
        val sumInit = context.nextTempValue(); val sumV = context.nextTempValue(); val sumB = context.nextTempValue()
        val attn = context.nextTempValue()
        val out = context.nextTempValue()

        val ops = mutableListOf(
            // Scale is applied to Q *before* the QK dot — scores·s == (q·s)@kᵀ exactly. This keeps the scale
            // constant at the STATIC query type (`[.., Sq, headDim]`) rather than a splat sized to the scores
            // shape `[.., Sq, Sk]`, whose Sk (key/cache length) may be dynamic (`?`) in KV-cache decode — a
            // dynamic-shape splat constant is invalid StableHLO. Also drops a full scores-sized dense constant
            // from static graphs.
            "$scaleC = stablehlo.constant dense<$scaleVal> : tensor<$elem>",
            "$scaleB = stablehlo.broadcast_in_dim $scaleC, dims = [] : (tensor<$elem>) -> $qType",
            "$qScaled = stablehlo.multiply ${operands[0]}, $scaleB : $qType",
        )
        // GQA: expose the head groups as a batching dim of Q — a static, copy-free reshape.
        val qForDot = if (gqa) {
            val qg = context.nextTempValue()
            ops += "$qg = stablehlo.reshape $qScaled : ($qType) -> $qWorkType"
            qg
        } else qScaled
        ops += "$scores = stablehlo.dot_general $qForDot, ${operands[1]}, ${batchClause}contracting_dims = [$contractQ] x [$contractK] : ($qWorkType, $kType) -> $scoresType"

        // Explicit additive mask (operands[3]) — e.g. a sliding-window+causal
        // mask the caller built and passed with causal=false. It already
        // encodes causality/window, so it takes priority over the built-in
        // iota causal path. Broadcast (trailing-aligned) to the scores shape
        // and add. Without this the masked layers run UNMASKED (attend to
        // future tokens) — correct only at position 0.
        var softmaxIn = scores   // scores are already scaled (scale folded into Q above)
        val maskOperand = operands.getOrNull(3)
        if (maskOperand != null) {
            val maskShape = node.inputs.getOrNull(3)?.shape ?: scoresShape
            val maskType = context.getValueType(maskOperand) ?: typeOf(maskShape)
            val maskBc = if (maskShape == scoresShape) {
                maskOperand
            } else {
                val mb = context.nextTempValue()
                // Trailing-aligned. Under GQA a rank-4 mask [b, 1|H, Sq, Sk] keeps its batch on
                // scores dim 0 and skips the nRep axis (dim 2): [0, 1, 3, 4].
                val dims = if (gqa && maskShape.size == 4) "0, 1, 3, 4" else {
                    val offset = scoresShape.size - maskShape.size
                    maskShape.indices.joinToString(", ") { (it + offset).toString() }
                }
                ops += "$mb = stablehlo.broadcast_in_dim $maskOperand, dims = [$dims] : ($maskType) -> $scoresType"
                mb
            }
            val masked = context.nextTempValue()
            ops += "$masked = stablehlo.add $scores, $maskBc : $scoresType"
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
            // Masked-fill with a large finite negative (not -inf): matches
            // MultiHeadAttention.buildSlidingCausalMask (-1e30) and avoids a -inf splat in the
            // masked-fill select, which can trip downstream greedy constant-folding.
            ops += "$ninf = stablehlo.constant dense<-1.000000e+30> : $scoresType"
            ops += "$maskAdd = stablehlo.select $keep, $zeros, $ninf : $scoresI1Type, $scoresType"
            ops += "$masked = stablehlo.add $scores, $maskAdd : $scoresType"
            softmaxIn = masked
        }

        // softmax(softmaxIn) over the key-length axis. When the scores shape is dynamic (the `?` key/cache dim
        // of KV-cache decode), the reduced max/sum must broadcast back to the dynamic scores shape. A static
        // `stablehlo.broadcast_in_dim` cannot target a dynamic shape, so we use `stablehlo.dynamic_broadcast_in_dim`
        // with a runtime `output_dimensions` operand (built once from the scores tensor via `get_dimension_size`).
        // Static graphs keep the original explicit `broadcast_in_dim` path (byte-for-byte unchanged).
        val dyn = scoresShape.hasDynamic()
        val shapeType = "tensor<${scoresShape.size}xi32>"
        val scoresShapeOperand: String = if (!dyn) "" else run {
            val parts = scoresShape.indices.map { d ->
                if (Dim.isStatic(scoresShape[d])) {
                    val c = context.nextTempValue()
                    ops += "$c = stablehlo.constant dense<${scoresShape[d]}> : tensor<1xi32>"
                    c
                } else {
                    val gd = context.nextTempValue(); val gr = context.nextTempValue()
                    ops += "$gd = stablehlo.get_dimension_size $scores, dim = $d : ($scoresType) -> tensor<i32>"
                    ops += "$gr = stablehlo.reshape $gd : (tensor<i32>) -> tensor<1xi32>"
                    gr
                }
            }
            val sh = context.nextTempValue()
            ops += "$sh = stablehlo.concatenate ${parts.joinToString(", ")}, dim = 0 : (${parts.joinToString(", ") { "tensor<1xi32>" }}) -> $shapeType"
            sh
        }
        fun broadcastBack(src: String, dst: String) {
            if (dyn) {
                ops += "$dst = stablehlo.dynamic_broadcast_in_dim $src, $scoresShapeOperand, dims = [$bcastDims] : ($reducedType, $shapeType) -> $scoresType"
            } else {
                ops += "$dst = stablehlo.broadcast_in_dim $src, dims = [$bcastDims] : ($reducedType) -> $scoresType"
            }
        }
        ops += "$maxInit = stablehlo.constant dense<${mapper.negInfBits(elem)}> : tensor<$elem>"
        ops += "$maxV = stablehlo.reduce($softmaxIn init: $maxInit) applies stablehlo.maximum across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType"
        broadcastBack(maxV, maxB)
        ops += "$shifted = stablehlo.subtract $softmaxIn, $maxB : $scoresType"
        ops += "$expV = stablehlo.exponential $shifted : $scoresType"
        ops += "$sumInit = stablehlo.constant dense<0.0> : tensor<$elem>"
        ops += "$sumV = stablehlo.reduce($expV init: $sumInit) applies stablehlo.add across dimensions = [$sdAxis] : ($scoresType, tensor<$elem>) -> $reducedType"
        broadcastBack(sumV, sumB)
        ops += "$attn = stablehlo.divide $expV, $sumB : $scoresType"
        if (gqa) {
            val outG = context.nextTempValue()
            ops += "$outG = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outWorkType"
            ops += "$out = stablehlo.reshape $outG : ($outWorkType) -> $outputType"
        } else {
            ops += "$out = stablehlo.dot_general $attn, ${operands[2]}, ${batchClause}contracting_dims = [$contractAttn] x [$contractV] : ($scoresType, $vType) -> $outputType"
        }
        ops.forEach { context.emitOperation(it) }
        return ConversionResult.Success(outputValueName = out, emittedOperations = ops)
    }
}
