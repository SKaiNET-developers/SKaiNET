package sk.ainet.lang.graph

import sk.ainet.context.ExecutionStats
import sk.ainet.context.ExecutionObserverRegistry
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

    private val observerRegistry = ExecutionObserverRegistry()

    /**
     * SKEEP-005 phase 2: this context does not build its ops, it wraps [baseOps]. So the schedule
     * is whatever the base ops run under — [sk.ainet.context.schedule.ScheduledOps] answers —
     * and [sk.ainet.context.schedule.Schedule.Sequential] for ops that know no schedule.
     */
    override val schedule: sk.ainet.context.schedule.Schedule
        get() = (baseOps as? sk.ainet.context.schedule.ScheduledOps)?.schedule
            ?: sk.ainet.context.schedule.Schedule.Sequential

    /**
     * A sibling context over `baseOps.withSchedule(schedule)` — fresh tape stack and trace
     * session, same phase, factory, hooks, stats, tape factory, graph and sink — mirroring
     * `DirectCpuExecutionContext.withSchedule`. When the base ops cannot be rescheduled the
     * request is a visible downgrade (the [sk.ainet.context.ExecutionContext] default emits
     * `TraceEvent.ScheduleDowngraded`), never a silent no-op.
     */
    override fun withSchedule(schedule: sk.ainet.context.schedule.Schedule): sk.ainet.context.ExecutionContext {
        if (schedule === this.schedule) return this
        val scheduled = baseOps as? sk.ainet.context.schedule.ScheduledOps ?: return super.withSchedule(schedule)
        return DefaultGraphExecutionContext(
            baseOps = scheduled.withSchedule(schedule),
            phase = phase,
            tensorDataFactory = tensorDataFactory,
            hooks = hooks,
            memoryInfo = memoryInfo,
            executionStats = executionStats,
            createTapeFactory = createTapeFactory,
            computeGraph = computeGraph,
            baseSink = baseSink,
        )
    }

    private val _tapes = DefaultTapeStack()
    override val tapeStack: TapeStack get() = _tapes

    private var lastTape: ExecutionTape? = null

    override val currentTape: ExecutionTape? get() = _tapes.currentTape

    override fun startRecording() {
        val tape = createTapeFactory(this)
        if (tape is DefaultExecutionTape) {
            tape.session = this.session
        } else if (tape is DefaultGradientTape) {
            tape.session = this.session
        }
        tape.startRecording()
        _tapes.pushTape(tape)
    }

    override fun stopRecording(): ExecutionTape? {
        val tape = _tapes.popTape()
        tape?.stopRecording()
        lastTape = tape
        return tape
    }

    /** Helper for internal use that returns the tape */
    @Deprecated("Use stopRecording() instead as it now returns the tape", ReplaceWith("stopRecording()"))
    public fun stopRecordingAndGet(): ExecutionTape? = stopRecording()

    @Suppress("UNCHECKED_CAST")
    override fun backward(targets: List<sk.ainet.lang.tensor.Tensor<*, *>>, sources: List<sk.ainet.lang.tensor.Tensor<*, *>>) {
        val tape = lastTape
        require(tape is sk.ainet.tape.GradientTape) { "No gradient tape available for backward pass. Ensure you recorded operations first." }
        tape.computeGradients(
            targets as List<sk.ainet.lang.tensor.Tensor<sk.ainet.lang.types.DType, Any?>>,
            sources as List<sk.ainet.lang.tensor.Tensor<sk.ainet.lang.types.DType, Any?>>
        )
    }

    override fun collectGarbage() { /* no-op */
    }

    override fun resetExecutionStats() { /* no-op */
    }

    private lateinit var _session: sk.ainet.lang.trace.TraceSession

    private val _ops: KspTensorOps by lazy {
        val dynamicSink = object : OpSink {
            override fun onOpExecuted(trace: sk.ainet.lang.trace.OpTrace) {
                val tape = tapeStack.currentTape
                if (tapeStack.isRecording() && tape != null) {
                    TapeSink(tape).onOpExecuted(trace)
                }
                baseSink.onOpExecuted(trace)
            }
        }
        val sharedSession = sk.ainet.lang.trace.TraceSession()
        _session = sharedSession
        KspTensorOps(baseOps, dynamicSink, sharedSession)
    }

    override val ops: KspTensorOps get() = _ops

    public val session: sk.ainet.lang.trace.TraceSession get() {
        _ops // trigger lazy init
        return _session
    }

    /**
     * Clear cached tensor references from the trace session.
     * Call between training batches to prevent memory accumulation.
     */
    public fun clearSession() {
        if (::_session.isInitialized) {
            _session.clear()
        }
    }

    override val observers: ExecutionObserverRegistry
        get() = observerRegistry

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
