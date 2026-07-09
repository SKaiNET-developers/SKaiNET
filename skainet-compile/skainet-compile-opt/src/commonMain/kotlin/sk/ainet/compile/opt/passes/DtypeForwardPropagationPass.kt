package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Forward dtype propagation — makes graph edges dtype-consistent so the
 * StableHLO emitter produces well-typed MLIR.
 *
 * Walks the graph in topological order and, for every node:
 *  1. sets each **input** spec's dtype to its producer's actual **output** dtype
 *     (an edge whose producer emits `bf16` but whose consumer input spec says
 *     `f32` renders the same SSA value with two types — malformed MLIR); and
 *  2. for dtype-**preserving** ops, sets the **output** spec dtype to the (now
 *     unified) input dtype, so the whole f32/bf16 island collapses to one dtype.
 *
 * This fixes bf16-native traces where reductions / normalizations were recorded
 * with a stale FP32 dtype while their producers emit bf16 (e.g. the Moonshine
 * encoder's LayerNorm reduce). It is a **no-op for uniformly-typed graphs**
 * (all-FP32 models are already edge-consistent, so nothing changes).
 *
 * When [targetFloatDtype] is set (e.g. `"BF16"`), every **float** source node's
 * output is coerced to it first, so a bf16-native model unifies to bf16 *end to
 * end* — including matmul activations, which the Torq NPU requires (bf16 weights
 * alone leave `f32 activation × bf16 weight` matmuls). Weights already at the
 * target dtype are a no-op; integer/bool sources (indices, masks) are left alone.
 * With [targetFloatDtype] null the pass only enforces edge-consistency.
 *
 * Left untouched:
 *  - **source** nodes' non-float dtypes (indices/masks): authoritative.
 *  - explicit **dtype-changing** ops (`convert`/`cast`/`quantize`/`dequantize`,
 *    `argmax`/`argmin`, comparisons): their output dtype is intentional.
 */
public class DtypeForwardPropagationPass(
    private val targetFloatDtype: String? = null,
) : GraphOptimizationPass {

    override val name: String = "dtype-forward-propagation"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        val topo = graph.getTopologicalOrder()
        val incoming = graph.edges.groupBy { it.destination.id }

        // Updated node per id; seeded with the originals, overwritten in topo order
        // so a consumer always reads its producer's already-updated output dtype.
        val updated: MutableMap<String, GraphNode> = graph.nodes.associateBy { it.id }.toMutableMap()
        var changed = false

        // Coerce float source nodes to the target dtype so the whole model unifies
        // to it (not just edges downstream of the weights).
        if (targetFloatDtype != null) {
            for (node in graph.nodes) {
                if (incoming[node.id]?.isNotEmpty() == true) continue // not a source
                val newOutputs = node.outputs.map {
                    if (isFloat(it.dtype) && it.dtype != targetFloatDtype) it.copy(dtype = targetFloatDtype) else it
                }
                if (newOutputs != node.outputs) {
                    updated[node.id] = node.copy(outputs = newOutputs)
                    changed = true
                }
            }
        }

        for (orig in topo) {
            val node = updated[orig.id] ?: continue
            val edges = incoming[orig.id].orEmpty()
            if (edges.isEmpty()) continue // source node — keep its declared dtype

            // (1) input spec dtype := producer output dtype
            val newInputs = node.inputs.toMutableList()
            for (e in edges) {
                val producer = updated[e.source.id] ?: continue
                val prodDtype = producer.outputs.getOrNull(e.sourceOutputIndex)?.dtype ?: continue
                val i = e.destinationInputIndex
                val spec = newInputs.getOrNull(i) ?: continue
                if (spec.dtype != prodDtype) newInputs[i] = spec.copy(dtype = prodDtype)
            }

            // (2) dtype-preserving ops inherit the (data) input dtype on outputs
            val inheritDtype = newInputs.firstOrNull()?.dtype
            val newOutputs = if (inheritDtype != null && !isDtypeChanging(node.operationName)) {
                node.outputs.map { if (it.dtype != inheritDtype) it.copy(dtype = inheritDtype) else it }
            } else {
                node.outputs
            }

            if (newInputs != node.inputs || newOutputs != node.outputs) {
                updated[orig.id] = node.copy(inputs = newInputs, outputs = newOutputs)
                changed = true
            }
        }

        if (!changed) return GraphOptimizationResult(graph, changed = false, diagnostics = diagnostics)

        // Rebuild the graph so edges reference the updated node instances (the
        // topo sort keys on node identity, so edges must point at the new nodes).
        val newGraph = DefaultComputeGraph()
        for (node in graph.nodes) newGraph.addNode(updated[node.id] ?: node)
        for (edge in graph.edges) {
            val src = updated[edge.source.id] ?: continue
            val dst = updated[edge.destination.id] ?: continue
            val spec = src.outputs.getOrNull(edge.sourceOutputIndex) ?: edge.tensorSpec
            newGraph.addEdge(edge.copy(source = src, destination = dst, tensorSpec = spec))
        }
        return GraphOptimizationResult(newGraph, changed = true, diagnostics = diagnostics)
    }

    private fun isFloat(dtype: String): Boolean = dtype.uppercase() in FLOAT_DTYPES

    private fun isDtypeChanging(op: String): Boolean {
        val n = op.lowercase()
        return n in DTYPE_CHANGING ||
            n.startsWith("convert") || n.startsWith("cast") ||
            n.startsWith("quant") || n.startsWith("dequant")
    }

    private companion object {
        val FLOAT_DTYPES: Set<String> = setOf(
            "F32", "FP32", "FLOAT32", "F16", "FP16", "FLOAT16", "BF16", "BFLOAT16", "F64", "FP64", "FLOAT64",
        )
        val DTYPE_CHANGING: Set<String> = setOf(
            "argmax", "argmin", "greater", "greaterequal", "less", "lessequal",
            "equal", "notequal", "compare", "logicaland", "logicalor", "logicalnot",
            "isnan", "isinf",
        )
    }
}
