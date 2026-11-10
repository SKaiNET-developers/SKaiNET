package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Folds dtype casts (`convert` / `cast` / `to`) that change no value (#1494).
 *
 * Two rewrites, both value-preserving:
 *
 * 1. **A cast after an exact widening.** In `A -> B -> C`, when every value of `A` is
 *    representable in `B` (FP16 or BF16 to FP32, any of them to FP64), the second cast reads
 *    from the original `A` tensor instead: rounding an exactly widened value gives the same
 *    result as rounding the value itself. `bf16 -> f32 -> f16` becomes `bf16 -> f16`. The
 *    widening cast is removed once nothing else reads it.
 * 2. **A cast to the dtype its input already has**, which rewrite 1 leaves behind for
 *    `f16 -> f32 -> f16` and which [DtypeForwardPropagationPass] can produce when it coerces a
 *    graph to one float dtype. The node is removed and its consumers read its input.
 *
 * This is what two adjacent mixed-precision boundaries need (SKEEP-008): each casts back to
 * the network dtype, so between them a tensor is widened only to be narrowed again.
 *
 * A *narrowing* followed by a widening (`f32 -> bf16 -> f32`) is never touched: that pair is
 * the rounding a mixed-precision layer asked for. Casts from or to integer dtypes are not
 * treated as exact (Int32 to FP32 is not) and are left alone, except by rewrite 2.
 */
public class CastFoldingPass : GraphOptimizationPass {

    override val name: String = "cast-folding"

    private class Link(
        val id: String,
        var sourceId: String,
        var sourceOutputIndex: Int,
        val destinationId: String,
        val destinationInputIndex: Int,
        var tensorSpec: TensorSpec,
        val metadata: Map<String, Any>,
    )

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        if (graph.nodes.none { isCast(it) }) return GraphOptimizationResult(graph, changed = false)

        val nodes = LinkedHashMap<String, GraphNode>()
        for (node in graph.nodes) nodes[node.id] = node
        val links = graph.edges.mapTo(mutableListOf()) {
            Link(
                it.id, it.source.id, it.sourceOutputIndex, it.destination.id,
                it.destinationInputIndex, it.tensorSpec, it.metadata,
            )
        }
        val diagnostics = mutableListOf<String>()

        // One rewrite per round keeps the bookkeeping simple; a round is linear in the graph
        // and the number of rounds is bounded by the number of cast nodes.
        while (foldOne(nodes, links, diagnostics)) Unit

        if (diagnostics.isEmpty()) return GraphOptimizationResult(graph, changed = false)

        val newGraph = DefaultComputeGraph()
        val added = HashMap<String, GraphNode>()
        for (node in nodes.values) added[node.id] = newGraph.addNode(node.copy())
        for (link in links) {
            val source = added[link.sourceId] ?: continue
            val destination = added[link.destinationId] ?: continue
            newGraph.addEdge(
                GraphEdge(
                    id = link.id,
                    source = source,
                    destination = destination,
                    sourceOutputIndex = link.sourceOutputIndex,
                    destinationInputIndex = link.destinationInputIndex,
                    tensorSpec = link.tensorSpec,
                    metadata = link.metadata,
                )
            )
        }
        return GraphOptimizationResult(newGraph, changed = true, diagnostics = diagnostics)
    }

    /** Applies the first rewrite that matches; `false` when none does. */
    private fun foldOne(
        nodes: MutableMap<String, GraphNode>,
        links: MutableList<Link>,
        diagnostics: MutableList<String>,
    ): Boolean {
        for (cast in nodes.values) {
            if (!isCast(cast)) continue
            val incoming = links.filter { it.destinationId == cast.id }
            if (incoming.size != 1) continue
            val input = incoming.single()
            val producer = nodes[input.sourceId] ?: continue
            val producerSpec = producer.outputs.getOrNull(input.sourceOutputIndex) ?: continue
            val from = canonical(producerSpec.dtype)
            val to = canonical(cast.outputs.firstOrNull()?.dtype ?: continue)

            // Rewrite 2: the cast changes nothing; its consumers read its input.
            if (from == to) {
                links.remove(input)
                for (out in links) {
                    if (out.sourceId != cast.id) continue
                    out.sourceId = producer.id
                    out.sourceOutputIndex = input.sourceOutputIndex
                    out.tensorSpec = producerSpec
                }
                nodes.remove(cast.id)
                diagnostics += "Removed ${cast.id}: cast from $from to the same dtype"
                return true
            }

            // Rewrite 1: the input is itself a cast, and an exact one; read through it.
            if (!isCast(producer)) continue
            val widenedFrom = links.filter { it.destinationId == producer.id }.singleOrNull() ?: continue
            val origin = nodes[widenedFrom.sourceId] ?: continue
            val originSpec = origin.outputs.getOrNull(widenedFrom.sourceOutputIndex) ?: continue
            val originType = canonical(originSpec.dtype)
            if (!isExactWidening(originType, from)) continue

            input.sourceId = origin.id
            input.sourceOutputIndex = widenedFrom.sourceOutputIndex
            input.tensorSpec = originSpec
            nodes[cast.id] = cast.copy(
                inputs = listOf(originSpec) + cast.inputs.drop(1),
            )
            diagnostics += "Folded ${producer.id} ($originType -> $from) into ${cast.id}: now $originType -> $to"
            if (links.none { it.sourceId == producer.id }) {
                links.remove(widenedFrom)
                nodes.remove(producer.id)
            }
            return true
        }
        return false
    }

    private fun isCast(node: GraphNode): Boolean = node.operationName.lowercase() in CAST_OPS

    private fun canonical(dtype: String): String = when (val upper = dtype.uppercase()) {
        "F16", "FLOAT16" -> "FP16"
        "BFLOAT16" -> "BF16"
        "F32", "FLOAT32" -> "FP32"
        "F64", "FLOAT64" -> "FP64"
        else -> upper
    }

    /** Every value of [from] is representable in [to]. Floating-point dtypes only. */
    private fun isExactWidening(from: String, to: String): Boolean = when (to) {
        "FP32" -> from == "FP16" || from == "BF16"
        "FP64" -> from == "FP16" || from == "BF16" || from == "FP32"
        else -> false
    }

    private companion object {
        val CAST_OPS: Set<String> = setOf("convert", "cast", "to")
    }
}
