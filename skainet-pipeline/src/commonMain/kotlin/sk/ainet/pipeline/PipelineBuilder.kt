package sk.ainet.pipeline

import kotlin.jvm.JvmInline

/**
 * DSL marker for pipeline builders.
 */
@DslMarker
public annotation class PipelineDsl

/**
 * Creates a new pipeline using the DSL.
 *
 * Example:
 * ```kotlin
 * val myPipeline = pipeline<String, Int>("parser") {
 *     node<String, List<String>>("tokenize") { input ->
 *         input.split(" ")
 *     }
 *     node<List<String>, Int>("count") { tokens ->
 *         tokens.size
 *     }
 * }
 * ```
 *
 * @param name Pipeline identifier
 * @param block DSL configuration block
 * @return Configured pipeline
 */
public fun <I, O> pipeline(
    name: String,
    block: PipelineBuilder<I, O>.() -> Unit
): Pipeline<I, O> {
    val builder = PipelineBuilder<I, O>(name)
    builder.block()
    return builder.build()
}

/**
 * Builder for constructing pipelines using a DSL.
 */
@PipelineDsl
public class PipelineBuilder<I, O>(private val name: String) {
    private val nodes = mutableMapOf<String, Node<*, *>>()
    private val edges = mutableListOf<Edge>()
    private val conditionalEdges = mutableListOf<ConditionalEdge<*>>()
    private var entryNode: String? = null
    private var exitNode: String? = null
    private var previousNode: String? = null

    /**
     * Define a processing node.
     *
     * Nodes are automatically chained in definition order unless
     * explicit edges are specified.
     *
     * @param name Unique node identifier
     * @param block Processing function
     * @return Reference to the created node
     */
    public fun <NI, NO> node(
        name: String,
        block: (NI) -> NO
    ): NodeRef<NO> {
        val node = FunctionNode(name, block)
        nodes[name] = node

        // Auto-chain from previous node
        previousNode?.let { prev ->
            edges.add(Edge(prev, name))
        }
        previousNode = name

        // First node is entry by default
        if (entryNode == null) {
            entryNode = name
        }

        return NodeRef(name)
    }

    /**
     * Define entry point (overrides auto-detection).
     */
    public fun entry(nodeName: String) {
        entryNode = nodeName
    }

    /**
     * Define exit point (overrides auto-detection).
     */
    public fun exit(nodeName: String) {
        exitNode = nodeName
    }

    /**
     * Define an explicit edge between nodes.
     *
     * Use this when you need non-linear flow or to override auto-chaining.
     */
    public fun edge(from: String, to: String) {
        edges.add(Edge(from, to))
    }

    /**
     * Define a conditional edge for dynamic routing.
     *
     * The router function receives the output of the source node
     * and returns the name of the target node.
     *
     * @param from Source node name
     * @param router Function that determines target node
     */
    public fun <T> conditionalEdge(
        from: String,
        router: (T) -> String
    ) {
        conditionalEdges.add(ConditionalEdge(from, router))
    }

    /**
     * Embed an existing pipeline as a node.
     *
     * @param name Node identifier for the embedded pipeline
     * @param pipeline The pipeline to embed
     * @return Reference to the created node
     */
    public fun <PI, PO> embed(
        name: String,
        pipeline: Pipeline<PI, PO>
    ): NodeRef<PO> {
        val node = PipelineNode(name, pipeline)
        nodes[name] = node

        previousNode?.let { prev ->
            edges.add(Edge(prev, name))
        }
        previousNode = name

        if (entryNode == null) {
            entryNode = name
        }

        return NodeRef(name)
    }

    /**
     * Build the pipeline.
     *
     * @return Configured pipeline
     * @throws IllegalStateException if pipeline is invalid
     */
    public fun build(): Pipeline<I, O> {
        val entry = entryNode
            ?: error("Pipeline '$name' must have at least one node")

        val exit = exitNode ?: previousNode
            ?: error("Pipeline '$name' must have at least one node")

        require(nodes.containsKey(entry)) {
            "Entry node '$entry' not found in pipeline '$name'"
        }
        require(nodes.containsKey(exit)) {
            "Exit node '$exit' not found in pipeline '$name'"
        }

        return Pipeline(
            name = name,
            nodes = nodes.toMap(),
            edges = edges.toList(),
            conditionalEdges = conditionalEdges.toList(),
            entryNode = entry,
            exitNode = exit
        )
    }
}

/**
 * Reference to a node, used for chaining in the DSL.
 */
@JvmInline
public value class NodeRef<T>(public val name: String)
