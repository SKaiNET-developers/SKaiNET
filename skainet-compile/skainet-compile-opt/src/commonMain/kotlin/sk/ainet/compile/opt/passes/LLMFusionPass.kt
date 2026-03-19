package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation

/**
 * LLM-specific operation fusion pass that recognizes and fuses common transformer patterns
 * into single compound operations that map to efficient fused kernels.
 *
 * Fusion patterns:
 *
 * 1. **RMSNorm**: `multiply(x,x) → mean → add(eps) → sqrt → rdiv(1.0) → multiply → multiply(weight)`
 *    → `fused_rms_norm` (single kernel, one pass over data)
 *
 * 2. **SwiGLU FFN**: `matmul(gate) → silu → multiply(with matmul(up)) → matmul(down)`
 *    → `fused_swiglu_ffn` (reduces memory traffic for intermediate activations)
 *
 * 3. **QKV merge**: Three matmul nodes sharing the same input (the norm output)
 *    → `fused_qkv_proj` (single batched matmul producing [Q|K|V], then split)
 *
 * This pass is designed to run **after** [TransposeEliminationPass] and
 * [SharedWeightDeduplicationPass] so it operates on a clean graph.
 */
public class LLMFusionPass : GraphOptimizationPass {
    override val name: String = "llm-fusion"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()

        // Build lookup structures
        val consumerMap = buildConsumerMap(graph)
        val producerMap = buildProducerMap(graph)
        val nodeById = graph.nodes.associateBy { it.id }

        val fusedNodeIds = mutableSetOf<String>()
        val replacements = mutableMapOf<String, List<GraphNode>>()
        // Maps each absorbed (non-anchor) node ID to its anchor node ID
        val absorbedToAnchor = mutableMapOf<String, String>()
        // Maps absorbed/anchor node IDs to their output offset within the fused node's combined outputs
        val outputOffsets = mutableMapOf<String, Int>()

        // --- Pattern 1: RMSNorm fusion ---
        for (node in graph.nodes) {
            if (node.id in fusedNodeIds) continue
            val chain = tryMatchRmsNorm(node, consumerMap, nodeById)
            if (chain != null) {
                val (chainNodes, epsValue) = chain
                val fusedOp = GenericOperation(
                    name = "fused_rms_norm",
                    parameters = mapOf(
                        "eps" to (epsValue ?: 1e-5f),
                        "fused_from" to chainNodes.map { it.operation.name }
                    ),
                    type = "fused"
                )
                val anchorNode = chainNodes.first()
                val lastNode = chainNodes.last()
                val fusedNode = anchorNode.copy(
                    operation = fusedOp,
                    outputs = lastNode.outputs
                )
                for (n in chainNodes) {
                    fusedNodeIds.add(n.id)
                    if (n.id != anchorNode.id) {
                        absorbedToAnchor[n.id] = anchorNode.id
                    }
                }
                replacements[anchorNode.id] = listOf(fusedNode)
                diagnostics.add(
                    "Fused RMSNorm: ${chainNodes.map { it.id }} → fused_rms_norm (${anchorNode.id})"
                )
            }
        }

        // --- Pattern 2: SwiGLU FFN fusion ---
        for (node in graph.nodes) {
            if (node.id in fusedNodeIds) continue
            val swiGlu = tryMatchSwiGlu(node, consumerMap, producerMap, nodeById)
            if (swiGlu != null) {
                val (gateMatmul, siluNode, upMatmul, mulNode, downMatmul) = swiGlu
                val fusedOp = GenericOperation(
                    name = "fused_swiglu_ffn",
                    parameters = mapOf(
                        "fused_from" to listOf("matmul_gate", "silu", "matmul_up", "multiply", "matmul_down")
                    ),
                    type = "fused"
                )
                val fusedNode = gateMatmul.copy(
                    operation = fusedOp,
                    outputs = downMatmul.outputs
                )
                val allIds = listOf(gateMatmul.id, siluNode.id, upMatmul.id, mulNode.id, downMatmul.id)
                fusedNodeIds.addAll(allIds)
                for (id in allIds) {
                    if (id != gateMatmul.id) {
                        absorbedToAnchor[id] = gateMatmul.id
                    }
                }
                replacements[gateMatmul.id] = listOf(fusedNode)
                diagnostics.add(
                    "Fused SwiGLU FFN: gate=${gateMatmul.id}, up=${upMatmul.id}, down=${downMatmul.id}"
                )
            }
        }

        // --- Pattern 3: QKV merge ---
        for (node in graph.nodes) {
            if (node.id in fusedNodeIds) continue
            val qkvTriple = tryMatchQKVProjections(node, consumerMap, nodeById)
            if (qkvTriple != null) {
                val (q, k, v) = qkvTriple
                val fusedOp = GenericOperation(
                    name = "fused_qkv_proj",
                    parameters = mapOf(
                        "fused_from" to listOf(q.operation.name, k.operation.name, v.operation.name),
                        "q_node" to q.id,
                        "k_node" to k.id,
                        "v_node" to v.id
                    ),
                    type = "fused"
                )
                // Fused QKV produces 3 outputs (Q, K, V)
                val combinedOutputs = q.outputs + k.outputs + v.outputs
                val fusedNode = q.copy(
                    operation = fusedOp,
                    outputs = combinedOutputs
                )
                fusedNodeIds.addAll(listOf(q.id, k.id, v.id))
                absorbedToAnchor[k.id] = q.id
                absorbedToAnchor[v.id] = q.id
                // Track output offsets for multi-output fused node
                outputOffsets[q.id] = 0
                outputOffsets[k.id] = q.outputs.size
                outputOffsets[v.id] = q.outputs.size + k.outputs.size
                replacements[q.id] = listOf(fusedNode)
                diagnostics.add(
                    "Fused QKV projections: q=${q.id}, k=${k.id}, v=${v.id} → fused_qkv_proj"
                )
            }
        }

        if (fusedNodeIds.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Rebuild graph
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            if (node.id in fusedNodeIds && node.id !in replacements) continue
            val replacement = replacements[node.id]
            if (replacement != null) {
                for (r in replacement) {
                    val copied = r.copy()
                    newGraph.addNode(copied)
                    nodeMap[copied.id] = copied
                }
            } else {
                val copied = node.copy()
                newGraph.addNode(copied)
                nodeMap[copied.id] = copied
            }
        }

        // Rewire edges
        for (edge in graph.edges) {
            val srcAbsorbed = edge.source.id in absorbedToAnchor
            val dstAbsorbed = edge.destination.id in absorbedToAnchor

            // Internal edges within the SAME fused group: drop
            if (srcAbsorbed && dstAbsorbed) {
                val srcAnchor = absorbedToAnchor[edge.source.id]!!
                val dstAnchor = absorbedToAnchor[edge.destination.id]!!
                if (srcAnchor == dstAnchor) continue
                // Cross-group edge between two absorbed nodes: rewire both ends
                val src = nodeMap[srcAnchor] ?: continue
                val dst = nodeMap[dstAnchor] ?: continue
                newGraph.addEdge(edge.copy(source = src, destination = dst))
                continue
            }
            // Edge from anchor to its own absorbed node: internal, drop
            if (edge.source.id in replacements && dstAbsorbed &&
                absorbedToAnchor[edge.destination.id] == edge.source.id) continue
            // Edge from absorbed node to its own anchor: internal, drop
            if (srcAbsorbed && edge.destination.id in replacements &&
                absorbedToAnchor[edge.source.id] == edge.destination.id) continue

            // Edge from absorbed node to external consumer: rewire source to the fused anchor
            // and adjust sourceOutputIndex for multi-output fused nodes (e.g., QKV)
            if (srcAbsorbed) {
                val anchorId = absorbedToAnchor[edge.source.id]!!
                val src = nodeMap[anchorId] ?: continue
                val dst = nodeMap[edge.destination.id] ?: continue
                val outOffset = outputOffsets[edge.source.id] ?: 0
                newGraph.addEdge(edge.copy(
                    source = src,
                    destination = dst,
                    sourceOutputIndex = edge.sourceOutputIndex + outOffset
                ))
                continue
            }
            // Edge to absorbed node from external: rewire destination to fused anchor
            if (dstAbsorbed) {
                val anchorId = absorbedToAnchor[edge.destination.id]!!
                val src = nodeMap[edge.source.id] ?: continue
                val dst = nodeMap[anchorId] ?: continue
                newGraph.addEdge(edge.copy(source = src, destination = dst))
                continue
            }

            val src = nodeMap[edge.source.id] ?: continue
            val dst = nodeMap[edge.destination.id] ?: continue
            // Apply output offset for anchor nodes with multi-output fused ops
            val outOffset = outputOffsets[edge.source.id] ?: 0
            newGraph.addEdge(edge.copy(
                source = src,
                destination = dst,
                sourceOutputIndex = edge.sourceOutputIndex + outOffset
            ))
        }

        return GraphOptimizationResult(
            graph = newGraph,
            changed = true,
            diagnostics = diagnostics
        )
    }

    // --- Pattern matchers ---

    /**
     * Tries to match a RMSNorm pattern starting from a multiply(x, x) node.
     * Returns the chain of nodes and the eps value, or null.
     */
    private fun tryMatchRmsNorm(
        startNode: GraphNode,
        consumers: Map<String, List<Pair<GraphNode, Int>>>,
        nodeById: Map<String, GraphNode>
    ): Pair<List<GraphNode>, Any?>? {
        // Pattern: multiply → mean → add(eps) → sqrt → rdiv/reciprocal → multiply → multiply(weight)
        if (startNode.operation.name != "multiply") return null

        val chain = mutableListOf(startNode)
        var current = startNode

        val expectedOps = listOf("mean", "add", "sqrt")
        for (expectedOp in expectedOps) {
            val next = singleConsumer(current, consumers) ?: return null
            if (!next.operation.name.startsWith(expectedOp)) return null
            chain.add(next)
            current = next
        }

        // After sqrt, expect rdiv or reciprocal, then multiply
        val afterSqrt = singleConsumer(current, consumers) ?: return null
        if (afterSqrt.operation.name !in setOf("rdiv", "reciprocal", "rsqrt", "divide")) return null
        chain.add(afterSqrt)
        current = afterSqrt

        // Expect multiply (rescale by normalized input)
        val rescale = singleConsumer(current, consumers) ?: return null
        if (rescale.operation.name != "multiply") return null
        chain.add(rescale)
        current = rescale

        // Expect multiply(weight) — the learned scale parameter
        val weightMul = singleConsumer(current, consumers) ?: return null
        if (weightMul.operation.name != "multiply") return null
        chain.add(weightMul)

        // Extract eps from the add node
        val addNode = chain[2] // the "add" node
        val eps = addNode.operation.parameters["value"]
            ?: addNode.operation.parameters["scalar"]
            ?: addNode.metadata["scalar"]

        return chain to eps
    }

    /**
     * Tries to match a SwiGLU pattern: gate_matmul → silu → multiply(with up_matmul) → down_matmul.
     */
    private fun tryMatchSwiGlu(
        startNode: GraphNode,
        consumers: Map<String, List<Pair<GraphNode, Int>>>,
        producers: Map<String, Map<Int, Pair<GraphNode, Int>>>,
        nodeById: Map<String, GraphNode>
    ): SwiGluMatch? {
        // Start from a matmul (gate projection)
        if (!isMatmulLike(startNode)) return null

        // gate_matmul → silu
        val siluNode = singleConsumer(startNode, consumers) ?: return null
        if (siluNode.operation.name != "silu") return null

        // silu → multiply
        val mulNode = singleConsumer(siluNode, consumers) ?: return null
        if (mulNode.operation.name != "multiply") return null

        // The multiply has two inputs: silu output and up_matmul output
        // Find the up_matmul: the other input to multiply that isn't the silu
        val mulInputs = producers[mulNode.id] ?: return null
        val upMatmul = mulInputs.values
            .map { (node, _) -> node }
            .firstOrNull { it.id != siluNode.id && isMatmulLike(it) }
            ?: return null

        // Verify gate and up matmuls share the same input (the norm output)
        val gateInput = producers[startNode.id]?.values?.firstOrNull()?.first
        val upInput = producers[upMatmul.id]?.values?.firstOrNull()?.first
        if (gateInput == null || upInput == null || gateInput.id != upInput.id) return null

        // multiply → down_matmul
        val downMatmul = singleConsumer(mulNode, consumers) ?: return null
        if (!isMatmulLike(downMatmul)) return null

        return SwiGluMatch(startNode, siluNode, upMatmul, mulNode, downMatmul)
    }

    /**
     * Tries to match 3 matmul nodes (Q, K, V projections) sharing the same input.
     */
    private fun tryMatchQKVProjections(
        sourceNode: GraphNode,
        consumers: Map<String, List<Pair<GraphNode, Int>>>,
        nodeById: Map<String, GraphNode>
    ): Triple<GraphNode, GraphNode, GraphNode>? {
        val nodeConsumers = consumers[sourceNode.id] ?: return null
        val matmulConsumers = nodeConsumers
            .map { (node, _) -> node }
            .filter { isMatmulLike(it) }
            .distinctBy { it.id }

        if (matmulConsumers.size < 3) return null

        // Heuristic: look for Q, K, V naming in node IDs or operation parameters
        val q = matmulConsumers.firstOrNull { "q_proj" in it.id || "attn_q" in it.id || "query" in it.id }
        val k = matmulConsumers.firstOrNull { "k_proj" in it.id || "attn_k" in it.id || "key" in it.id }
        val v = matmulConsumers.firstOrNull { "v_proj" in it.id || "attn_v" in it.id || "value" in it.id }

        if (q != null && k != null && v != null) {
            return Triple(q, k, v)
        }

        // Fallback: if exactly 3 matmuls consume the same node, assume QKV
        if (matmulConsumers.size == 3) {
            return Triple(matmulConsumers[0], matmulConsumers[1], matmulConsumers[2])
        }

        return null
    }

    // --- Helpers ---

    private fun singleConsumer(
        node: GraphNode,
        consumers: Map<String, List<Pair<GraphNode, Int>>>
    ): GraphNode? {
        val list = consumers[node.id] ?: return null
        return if (list.size == 1) list[0].first else null
    }

    private fun isMatmulLike(node: GraphNode): Boolean =
        node.operation.name in MATMUL_OPS

    private fun buildConsumerMap(graph: ComputeGraph): Map<String, List<Pair<GraphNode, Int>>> {
        val map = mutableMapOf<String, MutableList<Pair<GraphNode, Int>>>()
        for (edge in graph.edges) {
            map.getOrPut(edge.source.id) { mutableListOf() }
                .add(edge.destination to edge.destinationInputIndex)
        }
        return map
    }

    private fun buildProducerMap(graph: ComputeGraph): Map<String, MutableMap<Int, Pair<GraphNode, Int>>> {
        val map = mutableMapOf<String, MutableMap<Int, Pair<GraphNode, Int>>>()
        for (edge in graph.edges) {
            map.getOrPut(edge.destination.id) { mutableMapOf() }[edge.destinationInputIndex] =
                edge.source to edge.sourceOutputIndex
        }
        return map
    }

    private data class SwiGluMatch(
        val gateMatmul: GraphNode,
        val siluNode: GraphNode,
        val upMatmul: GraphNode,
        val mulNode: GraphNode,
        val downMatmul: GraphNode
    )

    private companion object {
        val MATMUL_OPS = setOf("matmul", "linear", "gemm", "batch_matmul")
    }
}
