package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Deduplicates parameter nodes that refer to the same underlying weight tensor.
 *
 * Many transformer architectures tie weights — for example, Llama and Apertus share
 * `token_embd.weight` with `output.weight`. After tracing, these appear as two separate
 * parameter nodes with identical shapes. This pass detects such duplicates and rewires
 * all consumers to read from a single canonical node, halving memory for the shared tensor.
 *
 * Detection heuristic:
 * - Both nodes are input/parameter nodes (no incoming edges)
 * - Both have the same output shape and dtype
 * - Their IDs suggest tying (configurable via [tiedWeightPatterns])
 *
 * When shapes match but names don't follow a known pattern, the pass conservatively
 * skips the pair — shape coincidence alone is not sufficient to guarantee weight tying.
 */
public class SharedWeightDeduplicationPass(
    /**
     * Pairs of ID substrings that indicate tied weights.
     * If parameter A's ID contains the first element and parameter B's ID contains the second,
     * and they have the same shape/dtype, they are considered tied.
     */
    private val tiedWeightPatterns: List<Pair<String, String>> = DEFAULT_TIED_PATTERNS
) : GraphOptimizationPass {
    override val name: String = "shared-weight-dedup"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()

        // Find all parameter/input nodes (nodes with no incoming edges)
        val inputNodeIds = graph.getInputNodes().map { it.id }.toSet()
        val paramNodes = graph.nodes.filter { node ->
            node.id in inputNodeIds && isParameterNode(node)
        }

        if (paramNodes.size < 2) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Build a signature for each parameter node: shape + dtype
        data class ParamSignature(val shape: List<Int>?, val dtype: String)

        val nodeSignatures = paramNodes.associateWith { node ->
            val output = node.outputs.firstOrNull()
            ParamSignature(output?.shape, output?.dtype ?: "unknown")
        }

        // Find tied pairs using the pattern list
        val redirectMap = mutableMapOf<String, String>() // nodeId to remove → canonical nodeId

        for (pattern in tiedWeightPatterns) {
            val candidates1 = paramNodes.filter { pattern.first in it.id }
            val candidates2 = paramNodes.filter { pattern.second in it.id }

            for (c1 in candidates1) {
                for (c2 in candidates2) {
                    if (c1.id == c2.id) continue
                    if (c1.id in redirectMap || c2.id in redirectMap) continue

                    val sig1 = nodeSignatures[c1] ?: continue
                    val sig2 = nodeSignatures[c2] ?: continue

                    if (sig1 == sig2 && sig1.shape != null) {
                        // Keep the one that appears first (embedding) as canonical
                        redirectMap[c2.id] = c1.id
                        diagnostics.add(
                            "Deduplicated: ${c2.id} → ${c1.id} " +
                                "(shape=${sig1.shape}, dtype=${sig1.dtype})"
                        )
                    }
                }
            }
        }

        if (redirectMap.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Rebuild graph, removing deduplicated nodes and rewiring edges
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            if (node.id in redirectMap) continue
            val copied = node.copy()
            newGraph.addNode(copied)
            nodeMap[copied.id] = copied
        }

        for (edge in graph.edges) {
            val effectiveSourceId = redirectMap[edge.source.id] ?: edge.source.id
            val src = nodeMap[effectiveSourceId] ?: continue
            val dst = nodeMap[edge.destination.id] ?: continue
            newGraph.addEdge(edge.copy(source = src, destination = dst))
        }

        return GraphOptimizationResult(
            graph = newGraph,
            changed = true,
            diagnostics = diagnostics
        )
    }

    private fun isParameterNode(node: GraphNode): Boolean {
        val op = node.operation
        // InputOperation with parameter kind, or nodes whose metadata marks them as parameters
        return op.name == "input" && (
            op.parameters["kind"] == "parameter" ||
                node.metadata["role"] == "parameter"
            ) ||
            // Also match traced parameter nodes
            node.metadata["role"] == "parameter"
    }

    public companion object {
        /**
         * Default patterns for detecting tied weights in common LLM architectures.
         */
        public val DEFAULT_TIED_PATTERNS: List<Pair<String, String>> = listOf(
            "token_embd" to "output",     // Llama, Apertus
            "embed_tokens" to "lm_head",  // HuggingFace naming
            "word_embeddings" to "output", // BERT-style
            "wte" to "lm_head",           // GPT-2 style
        )
    }
}
