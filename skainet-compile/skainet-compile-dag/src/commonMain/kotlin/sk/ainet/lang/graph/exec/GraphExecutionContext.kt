package sk.ainet.lang.graph.exec

import sk.ainet.context.ExecutionContext
import sk.ainet.context.TrainingExecutionContext
import sk.ainet.tape.ExecutionTape
import sk.ainet.tape.TapeStack
import sk.ainet.lang.tensor.ops.TensorOps

/**
 * Context for managing execution state, including mode switching,
 * device management, and memory management.
 */
public interface GraphExecutionContext : ExecutionContext, TrainingExecutionContext {

    public val baseOps: TensorOps

    public val createTapeFactory: (executionContext: GraphExecutionContext) -> ExecutionTape


    /**
     * Current execution tape (null if not recording)
     */
    public val currentTape: ExecutionTape?

    /**
     * Tape stack for nested execution contexts
     */
    public val tapeStack: TapeStack

    /**
     * Whether operations should be recorded. Overrides [ExecutionContext.isRecording] so modules
     * can route eager-vs-trace without depending on the graph module.
     */
    override val isRecording: Boolean get() = currentTape?.isRecording == true


    /**
     * Force garbage collection
     */
    public fun collectGarbage()

    /**
     * Reset execution statistics
     */
    public fun resetExecutionStats()
}