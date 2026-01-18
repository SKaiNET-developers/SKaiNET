package sk.ainet.pipeline

/**
 * A processing node in a pipeline graph.
 *
 * Nodes are stateless, reusable transformation units that process input
 * and produce output. They form the vertices of the pipeline graph.
 *
 * @param I Input type
 * @param O Output type
 */
public interface Node<I, O> {
    /**
     * Unique identifier for this node within the pipeline.
     */
    public val name: String

    /**
     * Execute the node's processing logic.
     *
     * @param input The input data to process
     * @return The processed output
     */
    public fun process(input: I): O
}

/**
 * A node that wraps a function.
 */
public class FunctionNode<I, O>(
    override val name: String,
    private val block: (I) -> O
) : Node<I, O> {
    override fun process(input: I): O = block(input)
}

/**
 * A node that wraps another pipeline, enabling composition.
 */
public class PipelineNode<I, O>(
    override val name: String,
    private val pipeline: Pipeline<I, O>
) : Node<I, O> {
    override fun process(input: I): O = pipeline.execute(input)
}
