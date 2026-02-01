package sk.ainet.context

/**
 * Wrapper that delegates to a base ExecutionContext but overrides the phase field.
 */
public class PhaseOverridingExecutionContext(
    private val base: ExecutionContext,
    override val phase: Phase
) : ExecutionContext {
    override val ops: sk.ainet.lang.tensor.ops.TensorOps get() = base.ops
    override val tensorDataFactory: sk.ainet.lang.tensor.data.TensorDataFactory get() = base.tensorDataFactory
    override val observers: ExecutionObserverRegistry get() = base.observers
    override val hooks: sk.ainet.lang.nn.hooks.ForwardHooks? get() = base.hooks
    override val memoryInfo: MemoryInfo get() = base.memoryInfo
    override val executionStats: ExecutionStats get() = base.executionStats
}

/**
 * Execute the given block with the phase set to TRAIN, using a delegating context wrapper.
 */
public inline fun <R> train(ctx: ExecutionContext, block: (ExecutionContext) -> R): R {
    val trainingCtx = PhaseOverridingExecutionContext(ctx, Phase.TRAIN)
    return block(trainingCtx)
}

/**
 * Execute the given block with the phase set to EVAL, using a delegating context wrapper.
 */
public inline fun <R> eval(ctx: ExecutionContext, block: (ExecutionContext) -> R): R {
    val evalCtx = PhaseOverridingExecutionContext(ctx, Phase.EVAL)
    return block(evalCtx)
}
