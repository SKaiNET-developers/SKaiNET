package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for memory-access / indexing operations.
 *
 * Today that's just `gather` and its framework aliases — the
 * critical path for LLM exports, where every transformer forward
 * pass begins with a token-id \u2192 embedding lookup. Without a
 * converter for `gather` / `embedding` / `index_select`, a traced
 * Llama / Mistral / Qwen / Gemma model fails at the very first
 * operation and never reaches the norms, activations, or attention
 * that the other P1 converters cover.
 *
 * The target lowering is the canonical `embedding(input_ids)`
 * shape: a 1-D index tensor indexing the leading dimension of a
 * 2-D embedding weight. Higher-rank gathers (attention-side index
 * gathers, multi-dim scatter/gather) can be added in follow-up PRs
 * once a traced model surfaces them; scoping this converter to the
 * LLM front-door case keeps review tight.
 *
 * Emitted shape:
 *
 * ```mlir
 * %out = stablehlo.gather(%weights, %indices)
 *     { dimension_numbers = #stablehlo.gather<
 *         offset_dims = [1],
 *         collapsed_slice_dims = [0],
 *         start_index_map = [0],
 *         index_vector_dim = 1>,
 *       slice_sizes = array<i64: 1, hidden_size>,
 *       indices_are_sorted = false }
 *     : (tensor<vocab_size x hidden_size x f32>, tensor<seq_len x i32>)
 *     -> tensor<seq_len x hidden_size x f32>
 * ```
 *
 * The `slice_sizes` vector is derived from the weight shape: a 1
 * along the gathered axis and the full extent of every other
 * dimension. `offset_dims`, `collapsed_slice_dims`, and
 * `start_index_map` are computed from the single gather axis.
 */
public class GatherOperationsConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(
        // "indexSelect" is the name the KSP tracing wrapper actually emits
        // (TensorOps.indexSelect); "index_select" is the framework-style
        // alias. Registry lookup is exact, so both must be listed (#1247).
        "gather", "embedding", "Embedding", "index_select", "indexSelect"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        return when (node.operation.name.lowercase()) {
            // "indexSelect".lowercase() is "indexselect", not "index_select" —
            // both lowercased spellings must be matched here (#1247).
            "gather", "embedding", "index_select", "indexselect" -> convertGather(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by GatherOperationsConverter"
            )
        }
    }

    private fun convertGather(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size < 2) {
            return ConversionResult.Failure(
                "Gather operation requires 2 operands (weights, indices), got ${operands.size}",
                "Unsupported gather arity for node ${node.id}"
            )
        }

        val weightSpec = node.inputs.getOrNull(0)
        val indicesSpec = node.inputs.getOrNull(1)
        val outputSpec = node.outputs.firstOrNull()

        val typeMapper = context.getTypeMapper()
        val weightType = weightSpec?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?xf32>"
        val indicesType = indicesSpec?.let { typeMapper.mapTensorType(it) } ?: "tensor<?xi32>"
        val outputType = outputSpec?.let { typeMapper.mapTensorType(it) } ?: "tensor<?x?xf32>"

        val weightShape = weightSpec?.shape ?: emptyList()
        val weightRank = weightShape.size
        val indicesRank = indicesSpec?.shape?.size ?: 1

        // Gather axis. Default to 0 (the conventional embedding-lookup
        // shape) and normalize negative axes against the weight rank.
        val rawAxis = node.operation.parameters["axis"] as? Int
            ?: node.operation.parameters["dim"] as? Int
            ?: 0
        val axis = when {
            weightRank == 0 -> 0
            rawAxis < 0 -> weightRank + rawAxis
            else -> rawAxis
        }.coerceIn(0, (weightRank - 1).coerceAtLeast(0))

        // offset_dims: the axes of the output that carry "the rest of
        // the row" — every weight axis except the gathered one, offset
        // by the indices rank (which sits at the beginning of the
        // output shape for a canonical gather).
        val offsetDims = (0 until weightRank)
            .filter { it != axis }
            .mapIndexed { i, _ -> indicesRank + i }
            .joinToString(", ")

        // collapsed_slice_dims: the axes of the weight that are
        // "picked" by the indices — just the gathered axis for this
        // single-axis case.
        val collapsedSliceDims = "$axis"

        // start_index_map: index `i` in the indices tensor maps to
        // start coordinate along the weight's gathered axis.
        val startIndexMap = "$axis"

        // index_vector_dim: the axis of the indices tensor that holds
        // the multi-dim coordinate. For a 1-D index tensor indexing a
        // single axis, this is the rank (i.e. one past the last dim),
        // following StableHLO convention that a trailing scalar
        // "implicit index vector" is allowed.
        val indexVectorDim = indicesRank

        // slice_sizes: a 1 along the gathered axis, the full extent
        // along every other axis.
        val sliceSizes = weightShape.mapIndexed { i, extent ->
            if (i == axis) 1 else extent
        }.joinToString(", ")

        val weightOperand = operands[0]
        val indicesOperand = operands[1]
        val resultValue = context.nextTempValue()
        // stablehlo.gather has no custom (pretty) assembly form — emit the
        // generic MLIR op form: "stablehlo.gather"(%operand, %indices) <{attrs}>.
        val gatherOp = "$resultValue = \"stablehlo.gather\"($weightOperand, $indicesOperand) " +
            "<{dimension_numbers = #stablehlo.gather<" +
            "offset_dims = [$offsetDims], " +
            "collapsed_slice_dims = [$collapsedSliceDims], " +
            "start_index_map = [$startIndexMap], " +
            "index_vector_dim = $indexVectorDim>, " +
            "slice_sizes = array<i64: $sliceSizes>, " +
            "indices_are_sorted = false}> " +
            ": ($weightType, $indicesType) -> $outputType"

        context.emitOperation(gatherOp)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(gatherOp)
        )
    }
}
