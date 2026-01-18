package sk.ainet.pipeline

/**
 * A type-safe processing pipeline represented as a directed graph.
 *
 * Pipelines consist of nodes (processing steps) connected by edges (data flow).
 * They support sequential, conditional, and parallel execution patterns.
 *
 * @param I Pipeline input type
 * @param O Pipeline output type
 */
public class Pipeline<I, O> internal constructor(
    /**
     * Name identifier for this pipeline.
     */
    public val name: String,

    /**
     * All nodes in this pipeline, keyed by name.
     */
    internal val nodes: Map<String, Node<*, *>>,

    /**
     * Direct edges connecting nodes.
     */
    internal val edges: List<Edge>,

    /**
     * Conditional edges for dynamic routing.
     */
    internal val conditionalEdges: List<ConditionalEdge<*>>,

    /**
     * Name of the entry node that receives pipeline input.
     */
    public val entryNode: String,

    /**
     * Name of the exit node that produces pipeline output.
     */
    public val exitNode: String
) {
    /**
     * Execute the pipeline with the given input.
     *
     * @param input The input data
     * @return The pipeline output
     */
    public fun execute(input: I): O {
        return PipelineExecutor.execute(this, input)
    }

    /**
     * Compose this pipeline with another, creating a sequential chain.
     *
     * @param other The pipeline to execute after this one
     * @return A new pipeline that executes this then other
     */
    public fun <R> then(other: Pipeline<O, R>): Pipeline<I, R> {
        return PipelineComposer.sequence(this, other)
    }

    /**
     * Get a node by name.
     */
    public fun getNode(name: String): Node<*, *>? = nodes[name]

    /**
     * Check if a node exists.
     */
    public fun hasNode(name: String): Boolean = nodes.containsKey(name)

    /**
     * Get all node names.
     */
    public fun nodeNames(): Set<String> = nodes.keys

    /**
     * Find outgoing edges from a node.
     */
    internal fun outgoingEdges(nodeName: String): List<Edge> =
        edges.filter { it.from == nodeName }

    /**
     * Find conditional edge from a node, if any.
     */
    internal fun conditionalEdgeFrom(nodeName: String): ConditionalEdge<*>? =
        conditionalEdges.find { it.from == nodeName }
}

/**
 * Executes pipeline graphs.
 */
internal object PipelineExecutor {

    @Suppress("UNCHECKED_CAST")
    fun <I, O> execute(pipeline: Pipeline<I, O>, input: I): O {
        var currentNode = pipeline.entryNode
        var currentData: Any? = input

        while (true) {
            val node = pipeline.nodes[currentNode]
                ?: error("Node '$currentNode' not found in pipeline '${pipeline.name}'")

            // Execute the node
            currentData = (node as Node<Any?, Any?>).process(currentData)

            // Check if we've reached the exit
            if (currentNode == pipeline.exitNode) {
                return currentData as O
            }

            // Find next node
            currentNode = findNextNode(pipeline, currentNode, currentData)
                ?: error("No outgoing edge from node '$currentNode' in pipeline '${pipeline.name}'")
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun findNextNode(
        pipeline: Pipeline<*, *>,
        current: String,
        data: Any?
    ): String? {
        // Check conditional edges first
        val conditional = pipeline.conditionalEdgeFrom(current)
        if (conditional != null) {
            return (conditional.router as (Any?) -> String)(data)
        }

        // Fall back to direct edges
        val outgoing = pipeline.outgoingEdges(current)
        return outgoing.firstOrNull()?.to
    }
}

/**
 * Composes pipelines together.
 */
internal object PipelineComposer {

    fun <I, M, O> sequence(first: Pipeline<I, M>, second: Pipeline<M, O>): Pipeline<I, O> {
        val combinedName = "${first.name}_then_${second.name}"

        // Create bridge node that executes second pipeline
        val bridgeNode = PipelineNode("__bridge__", second)

        // Combine nodes
        val combinedNodes = first.nodes.toMutableMap()
        combinedNodes["__bridge__"] = bridgeNode

        // Combine edges, connecting first's exit to bridge
        val combinedEdges = first.edges.toMutableList()
        combinedEdges.add(Edge(first.exitNode, "__bridge__"))

        return Pipeline(
            name = combinedName,
            nodes = combinedNodes,
            edges = combinedEdges,
            conditionalEdges = first.conditionalEdges,
            entryNode = first.entryNode,
            exitNode = "__bridge__"
        )
    }
}
