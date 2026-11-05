package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.trace.PackedConstantHandling

/**
 * Folds a block-quantized `Linear` into one `block_quant_matmul` node that keeps the weight packed.
 *
 * Matches a weight node carrying packed Q8_0 or Q4_0 storage (traced with
 * [PackedConstantHandling.KEEP_PACKED]) that feeds a matmul as `x · Wᵀ`, in either traced form:
 * ```
 *   weight[N, K] → transpose → matmul(x, ·)
 *   weight[N, K] → matmul(x, ·, transposeB = true)     (after TransposeEliminationPass)
 * ```
 * and replaces the three (or two) nodes with
 * ```
 *   block_quant_matmul(x)   parameters: weight = the PackedBlockStorage, weight_name
 * ```
 * The StableHLO emitter lowers that node to a dequantize-then-`batch_matmul` linalg chain with the
 * codes and per-block scales inline, the form an NPU compiler can match to a block-quantized kernel.
 *
 * Only an exclusive weight is folded: one that has other consumers (a tied embedding, say) stays as
 * it is, and so does one in another encoding, in kernel block order, or whose `K` is not a multiple
 * of the block size. Each skip is reported in the diagnostics, because a packed weight left behind
 * makes the emitter fail.
 *
 * A target opts in by registering the pass:
 * ```kotlin
 * TargetOptimizers.registerDagPasses("torq") { listOf(BlockQuantMatmulFusionPass()) }
 * ```
 */
public class BlockQuantMatmulFusionPass : GraphOptimizationPass {

    override val name: String = "block-quant-matmul-fusion"

    private data class Match(val weight: GraphNode, val transpose: GraphNode?, val matmul: GraphNode, val storage: PackedBlockStorage)

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        val consumers = graph.edges.groupBy { it.source.id }

        val matches = mutableListOf<Match>()
        for (weight in graph.nodes) {
            val storage = weight.operation.parameters[PackedConstantHandling.PACKED_WEIGHT_PARAM] as? PackedBlockStorage ?: continue
            val reason = ineligibility(storage)
            if (reason != null) {
                diagnostics += "BLOCK_QUANT_SKIPPED weight=${weight.id}: $reason"
                continue
            }
            val match = matchConsumer(weight, storage, consumers)
            if (match == null) {
                diagnostics += "BLOCK_QUANT_SKIPPED weight=${weight.id}: not the only input of a matmul as x · Wᵀ"
                continue
            }
            matches += match
        }
        if (matches.isEmpty()) return GraphOptimizationResult(graph, changed = false, diagnostics = diagnostics)

        val removed = matches.flatMap { listOfNotNull(it.weight.id, it.transpose?.id) }.toSet()
        val fused = matches.associate { m ->
            val spec = m.weight.outputs.firstOrNull()
            m.matmul.id to GraphNode(
                id = m.matmul.id,
                operation = GenericOperation(
                    name = OP_NAME,
                    parameters = mapOf(WEIGHT to m.storage, WEIGHT_NAME to (spec?.name ?: m.weight.id)),
                    type = "linalg",
                ),
                inputs = m.matmul.inputs.take(1),
                outputs = m.matmul.outputs,
                metadata = m.matmul.metadata,
            )
        }

        val rebuilt = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()
        for (node in graph.nodes) {
            if (node.id in removed) continue
            val toAdd = fused[node.id] ?: node
            rebuilt.addNode(toAdd)
            nodeMap[toAdd.id] = toAdd
        }
        for (edge in graph.edges) {
            if (edge.source.id in removed) continue
            // The weight operand of a fused matmul is gone with its producer.
            if (edge.destination.id in fused && edge.destinationInputIndex != 0) continue
            val src = nodeMap[edge.source.id] ?: continue
            val dst = nodeMap[edge.destination.id] ?: continue
            rebuilt.addEdge(GraphEdge(edge.id, src, dst, edge.sourceOutputIndex, edge.destinationInputIndex, edge.tensorSpec))
        }

        for (m in matches) {
            diagnostics += "BLOCK_QUANT_FUSED node=${m.matmul.id} weight=${m.weight.id} encoding=${m.storage.encoding.name} shape=${m.storage.shape.dimensions.toList()}"
        }
        return GraphOptimizationResult(rebuilt, changed = true, diagnostics = diagnostics)
    }

    private fun ineligibility(storage: PackedBlockStorage): String? = when {
        storage.encoding != TensorEncoding.Q8_0 && storage.encoding != TensorEncoding.Q4_0 ->
            "encoding ${storage.encoding.name} is not Q8_0 or Q4_0"
        storage.blockOrder != BlockOrder.ROW_MAJOR -> "blocks are in ${storage.blockOrder} order, not ROW_MAJOR"
        storage.shape.rank != 2 -> "shape ${storage.shape.dimensions.toList()} is not [out, in]"
        storage.shape[1] % storage.blockSize != 0 -> "in = ${storage.shape[1]} is not a multiple of ${storage.blockSize}"
        else -> null
    }

    private fun matchConsumer(weight: GraphNode, storage: PackedBlockStorage, consumers: Map<String, List<GraphEdge>>): Match? {
        val out = consumers[weight.id]?.singleOrNull() ?: return null
        val next = out.destination
        if (isMatmul(next)) {
            val transposed = next.operation.parameters["transposeB"] == true
            return if (transposed && out.destinationInputIndex == 1) Match(weight, null, next, storage) else null
        }
        if (!isPlainTranspose(next)) return null
        val mm = consumers[next.id]?.singleOrNull() ?: return null
        if (!isMatmul(mm.destination) || mm.destinationInputIndex != 1) return null
        if (mm.destination.operation.parameters["transposeA"] == true || mm.destination.operation.parameters["transposeB"] == true) return null
        return Match(weight, next, mm.destination, storage)
    }

    private fun isMatmul(node: GraphNode): Boolean = node.operation.name.lowercase() == "matmul"

    /** A transpose that swaps the two axes of a rank-2 tensor — the only one `x · Wᵀ` produces. */
    private fun isPlainTranspose(node: GraphNode): Boolean {
        if (node.operation.name.lowercase() != "transpose") return false
        val perm = (node.operation.parameters["permutation"] ?: node.operation.parameters["perm"] ?: node.operation.parameters["axes"]) as? List<*>
        return perm == null || perm.map { it.toString() } == listOf("1", "0")
    }

    public companion object {
        /** Op name of the fused node; the StableHLO emitter's block-quant converter lowers it. */
        public const val OP_NAME: String = "block_quant_matmul"
        /** Parameter holding the weight's [PackedBlockStorage], logical shape `[out, in]`. */
        public const val WEIGHT: String = "weight"
        /** Parameter holding the weight tensor's name, for comments and diagnostics. */
        public const val WEIGHT_NAME: String = "weight_name"
    }
}
