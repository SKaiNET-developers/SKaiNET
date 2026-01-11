package sk.ainet.lang.graph

import sk.ainet.context.ExecutionStats
import sk.ainet.context.MemoryInfo
import sk.ainet.context.Phase
import sk.ainet.lang.graph.exec.GraphExecutionContext
import sk.ainet.lang.tensor.ops.KspTensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.trace.CompositeSink
import sk.ainet.lang.trace.NoOpSink
import sk.ainet.lang.trace.OpSink
import sk.ainet.lang.trace.TapeSink
import sk.ainet.lang.nn.hooks.ForwardHooks
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.tape.ExecutionTape
import sk.ainet.tape.TapeStack

public class DefaultGraphExecutionContext(
    override val baseOps: TensorOps = VoidTensorOps(),
    override val phase: Phase = Phase.EVAL,
    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory(),
    override val hooks: ForwardHooks? = null,
    override val memoryInfo: MemoryInfo = MemoryInfo.getEmptyInfo(),
    override val executionStats: ExecutionStats = ExecutionStats(),
    override val createTapeFactory: (GraphExecutionContext) -> ExecutionTape =
        { _ -> DefaultExecutionTape() },

    /** Optional compute graph used by GraphSink presets. */
    public val computeGraph: ComputeGraph? = null,
    /**
     * Optional base sink configured at construction time. For presets we provide NoOp/Tape/Graph/Composite.
     * Dynamic TapeSink for the current tape (when recording) will be appended automatically.
     */
    private val baseSink: OpSink = NoOpSink,
    ) : GraphExecutionContext {

    private val _tapes = DefaultTapeStack()
    override val tapeStack: TapeStack get() = _tapes


    override val currentTape: ExecutionTape? get() = _tapes.currentTape

    public fun startRecording() {
        val tape = createTapeFactory(this)
        tape.startRecording()
        _tapes.pushTape(tape)
    }

    public fun stopRecording(): ExecutionTape? {
        val tape = _tapes.popTape()
        tape?.stopRecording()
        return tape
    }

    override fun collectGarbage() { /* no-op */
    }

    override fun resetExecutionStats() { /* no-op */
    }

    override val ops: TensorOps
        get() {
            val tape = currentTape
            val dynamicSink: OpSink = tape?.let {
                // Attach TapeSink only when recording and either in TRAIN phase or this is not a gradient tape.
                // This avoids unnecessary autograd/tape overhead during evaluation runs (FR2 guard).
                val allowRecording = phase == Phase.TRAIN || it !is DefaultGradientTape
                if (it.isRecording && allowRecording && it is DefaultExecutionTape) CompositeSink(listOf(baseSink, TapeSink(it))) else baseSink
            } ?: baseSink

            // Always expose KspTensorOps to avoid branching in hot path
            // If we have a DefaultExecutionTape, use its session for stability (FR7)
            val session = (tape as? DefaultExecutionTape)?.session ?: sk.ainet.lang.trace.TraceSession()
            return KspTensorOps(baseOps, dynamicSink, session)
        }

    /** Convenience helper to record within a block and return the produced tape (and keep existing graph). */
    public inline fun <R> record(block: DefaultGraphExecutionContext.() -> R): Pair<ExecutionTape?, R> {
        startRecording()
        return try {
            val result = this.block()
            stopRecording() to result
        } finally {
            if (isRecording) stopRecording()
        }
    }

    public companion object {
        /** Eager-only: no recording. */
        public fun eager(
            baseOps: TensorOps = VoidTensorOps(),
        ): DefaultGraphExecutionContext = DefaultGraphExecutionContext(
            baseOps = baseOps,
            baseSink = NoOpSink
        )

        /** Tape-only preset: tape is created on startRecording(); traces are appended via TapeSink. */
        public fun tape(
            baseOps: TensorOps = VoidTensorOps(),
            tapeFactory: (GraphExecutionContext) -> ExecutionTape = { _ -> DefaultExecutionTape() }
        ): DefaultGraphExecutionContext = DefaultGraphExecutionContext(
            baseOps = baseOps,
            createTapeFactory = tapeFactory,
            baseSink = NoOpSink // TapeSink is attached dynamically when recording
        )

        /** Graph-only preset: build graph online using GraphSink. */
        public fun graph(
            baseOps: TensorOps = VoidTensorOps(),
            graph: ComputeGraph = DefaultComputeGraph()
        ): DefaultGraphExecutionContext = DefaultGraphExecutionContext(
            baseOps = baseOps,
            computeGraph = graph,
            baseSink = sk.ainet.lang.trace.GraphSink(graph)
        )

        /** Composite preset: graph online; when recording also append to tape. */
        public fun tapeAndGraph(
            baseOps: TensorOps = VoidTensorOps(),
            graph: ComputeGraph = DefaultComputeGraph(),
            tapeFactory: (GraphExecutionContext) -> ExecutionTape = { _ -> DefaultExecutionTape() }
        ): DefaultGraphExecutionContext = DefaultGraphExecutionContext(
            baseOps = baseOps,
            createTapeFactory = tapeFactory,
            computeGraph = graph,
            baseSink = sk.ainet.lang.trace.GraphSink(graph)
        )
    }

}
