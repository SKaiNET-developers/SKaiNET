package sk.ainet.pipeline

/**
 * A directed edge connecting two nodes in a pipeline graph.
 *
 * Edges define the data flow between nodes. They can optionally
 * include a transformation function applied during data transfer.
 */
public data class Edge(
    /**
     * Name of the source node.
     */
    public val from: String,

    /**
     * Name of the target node.
     */
    public val to: String
)

/**
 * A conditional edge that routes data based on runtime conditions.
 *
 * The router function examines the input data and returns the name
 * of the target node to route to.
 *
 * @param I Input type that the router examines
 */
public class ConditionalEdge<I>(
    /**
     * Name of the source node.
     */
    public val from: String,

    /**
     * Function that determines the target node based on input data.
     */
    public val router: (I) -> String
)
