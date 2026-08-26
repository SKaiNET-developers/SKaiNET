package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.ResolvedComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.blockOrder
import sk.ainet.lang.tensor.ops.tensorEncoding
import sk.ainet.lang.tensor.ops.withBlockOrder
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The first real layout decision on the compile path (#1180): rank-2 block-quantized weights are
 * assigned kernel-feed block order (`INPUT_BLOCK_MAJOR`) for [target], mirroring the decision the
 * eager side's form resolver already makes at load — packed matmul kernels read input-block-major,
 * and a weight delivered in that order has nothing to convert on first use (#1120).
 *
 * Decision rules, deliberately narrow:
 * - only block-quantized encodings whose kernels feed them (the GGML block formats);
 * - only rank-2 specs (feed order is defined relative to an `[out, in]` weight);
 * - never overrides an order the tape already carried — the loader's fact outranks a preference.
 *
 * The pass stamps `backendAssignment` metadata on the nodes it touched, which
 * `ResolvedComputeGraph.backendAssignment` surfaces — so the pre-built seams stop returning null
 * exactly where a decision was actually made, and nowhere else.
 *
 * Mechanically target-parameterized rather than registry-registered: `HloGenerator` runs it as a
 * core pass of `dagPipelineFor(target, …)` whenever a target is named, and target-specific
 * optimizers can still contribute their own passes through [sk.ainet.compile.opt.TargetOptimizers].
 */
public class LayoutAssignmentPass(private val target: String) : GraphOptimizationPass {

    override val name: String = "layout-assignment($target)"

    public companion object {
        /** Block-quantized encodings whose matmul kernels read input-block-major (#973/#1120). */
        private val FEED_ORDERED = setOf<TensorEncoding>(
            TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
            TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0,
        )
    }

    private fun decide(spec: TensorSpec): TensorSpec? {
        val encoding = spec.tensorEncoding ?: return null
        if (encoding !in FEED_ORDERED) return null
        if (spec.shape?.size != 2) return null
        if (spec.blockOrder != null) return null // the tape's fact outranks this preference
        return spec.withBlockOrder("INPUT_BLOCK_MAJOR")
    }

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        var changed = false
        val newNodes = graph.nodes.map { node ->
            var touched = false
            val outputs = node.outputs.map { spec -> decide(spec)?.also { touched = true } ?: spec }
            val inputs = node.inputs.map { spec -> decide(spec)?.also { touched = true } ?: spec }
            if (!touched) return@map node
            changed = true
            node.copy(
                inputs = inputs,
                outputs = outputs,
                metadata = node.metadata + (ResolvedComputeGraph.BACKEND_ASSIGNMENT_METADATA_KEY to target),
            )
        }
        if (!changed) return GraphOptimizationResult(graph, changed = false)

        val byId = newNodes.associateBy { it.id }
        val newGraph = DefaultComputeGraph()
        for (node in newNodes) newGraph.addNode(node)
        for (edge in graph.edges) {
            val src = byId.getValue(edge.source.id)
            val dst = byId.getValue(edge.destination.id)
            val spec = decide(edge.tensorSpec) ?: edge.tensorSpec
            newGraph.addEdge(edge.copy(source = src, destination = dst, tensorSpec = spec))
        }
        return GraphOptimizationResult(newGraph, changed = true)
    }
}
