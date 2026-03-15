package sk.ainet.benchmark

import sk.ainet.context.ExecutionContext

@DslMarker
public annotation class BenchmarkDsl

public enum class BenchmarkMetric {
    LATENCY,
    MEMORY
}

public data class BenchmarkSuite(
    val name: String,
    val description: String?,
    val metrics: Set<BenchmarkMetric>,
    val contextFactory: () -> ExecutionContext,
    val cases: List<BenchmarkCase>
)

public data class BenchmarkCase(
    val name: String,
    val description: String?,
    val warmupIterations: Int,
    val iterations: Int,
    val setup: (BenchmarkExecutionScope.() -> Unit)?,
    val execute: BenchmarkExecutionScope.() -> Unit,
    val teardown: (BenchmarkExecutionScope.() -> Unit)?
)

@BenchmarkDsl
public fun benchmarkSuite(
    name: String,
    block: BenchmarkSuiteBuilder.() -> Unit
): BenchmarkSuite {
    val builder = BenchmarkSuiteBuilder(name)
    builder.block()
    return builder.build()
}

@BenchmarkDsl
public class BenchmarkSuiteBuilder internal constructor(
    private val name: String
) {
    private var description: String? = null
    private var contextFactory: (() -> ExecutionContext)? = null
    private val metrics: MutableSet<BenchmarkMetric> = mutableSetOf(BenchmarkMetric.LATENCY)
    private val cases: MutableList<BenchmarkCaseBuilder> = mutableListOf()

    public fun description(text: String) {
        description = text
    }

    public fun context(factory: () -> ExecutionContext) {
        contextFactory = factory
    }

    public fun metrics(block: BenchmarkMetricsBuilder.() -> Unit) {
        metrics.clear()
        val builder = BenchmarkMetricsBuilder(metrics)
        builder.block()
        if (metrics.isEmpty()) {
            metrics += BenchmarkMetric.LATENCY
        }
    }

    @BenchmarkDsl
    public fun case(name: String, block: BenchmarkCaseBuilder.() -> Unit) {
        val builder = BenchmarkCaseBuilder(name)
        builder.block()
        cases += builder
    }

    internal fun build(): BenchmarkSuite {
        val ctxFactory = contextFactory
            ?: throw IllegalStateException("Benchmark suite \"$name\" must define a context { ... } block.")
        if (cases.isEmpty()) {
            throw IllegalStateException("Benchmark suite \"$name\" must define at least one case { ... }")
        }
        val builtCases = cases.map { it.build() }
        return BenchmarkSuite(
            name = name,
            description = description,
            metrics = metrics.toSet(),
            contextFactory = ctxFactory,
            cases = builtCases
        )
    }
}

@BenchmarkDsl
public class BenchmarkMetricsBuilder internal constructor(
    private val target: MutableSet<BenchmarkMetric>
) {
    public fun latency() {
        target += BenchmarkMetric.LATENCY
    }

    public fun memory() {
        target += BenchmarkMetric.MEMORY
    }
}

@BenchmarkDsl
public class BenchmarkCaseBuilder internal constructor(
    private val name: String
) {
    private var description: String? = null
    private var warmupIterations: Int = 3
    private var iterations: Int = 10
    private var setupBlock: (BenchmarkExecutionScope.() -> Unit)? = null
    private var executeBlock: (BenchmarkExecutionScope.() -> Unit)? = null
    private var teardownBlock: (BenchmarkExecutionScope.() -> Unit)? = null

    public fun description(text: String) {
        description = text
    }

    public fun warmup(iterations: Int) {
        require(iterations >= 0) { "Warmup iterations must be >= 0" }
        warmupIterations = iterations
    }

    public fun iterations(iterations: Int) {
        require(iterations > 0) { "Iterations must be > 0" }
        this.iterations = iterations
    }

    public fun setup(block: BenchmarkExecutionScope.() -> Unit) {
        setupBlock = block
    }

    public fun run(block: BenchmarkExecutionScope.() -> Unit) {
        executeBlock = block
    }

    public fun teardown(block: BenchmarkExecutionScope.() -> Unit) {
        teardownBlock = block
    }

    internal fun build(): BenchmarkCase {
        val execute = executeBlock
            ?: throw IllegalStateException("Benchmark case \"$name\" must define run { ... }")
        return BenchmarkCase(
            name = name,
            description = description,
            warmupIterations = warmupIterations,
            iterations = iterations,
            setup = setupBlock,
            execute = execute,
            teardown = teardownBlock
        )
    }
}
