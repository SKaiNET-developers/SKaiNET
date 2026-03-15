package sk.ainet.benchmark

import sk.ainet.context.ExecutionContext
import sk.ainet.context.ExecutionObserver
import sk.ainet.context.ResettableExecutionObserver
import sk.ainet.context.observers.LatencyExecutionObserver
import sk.ainet.context.observers.LatencyMeasurement
import sk.ainet.context.observers.MemorySample
import sk.ainet.context.observers.MemorySnapshotObserver
import kotlin.time.Duration
import kotlin.time.Duration.Companion.ZERO
import kotlin.time.measureTime

public class BenchmarkExecutionScope internal constructor(
    public val context: ExecutionContext
) {
    private val storage: MutableMap<String, Any?> = mutableMapOf()

    public fun <T : Any> remember(key: String, value: T) {
        storage[key] = value
    }

    @Suppress("UNCHECKED_CAST")
    public fun <T : Any> recall(key: String): T =
        storage[key] as? T
            ?: throw IllegalStateException("No value stored for key \"$key\" in benchmark scope.")

    public fun clearStorage() {
        storage.clear()
    }
}

public data class LatencySummary(
    val measurements: List<LatencyMeasurement>,
    val totalDuration: Duration,
    val averageDuration: Duration,
    val count: Int,
    val totalsByOp: Map<String, Duration>
)

public data class MemorySummary(
    val samples: List<MemorySample>,
    val peakUsedBytes: Long,
    val lastUsedBytes: Long,
    val usagePercent: Double?
)

public data class BenchmarkCaseResult(
    val name: String,
    val description: String?,
    val warmupIterations: Int,
    val iterations: Int,
    val totalDuration: Duration,
    val durationPerIteration: Duration,
    val latency: LatencySummary?,
    val memory: MemorySummary?
)

public data class BenchmarkSuiteResult(
    val suiteName: String,
    val description: String?,
    val metrics: Set<BenchmarkMetric>,
    val cases: List<BenchmarkCaseResult>
) {
    public fun prettyPrint(): String {
        val sb = StringBuilder()
        sb.appendLine("Benchmark suite: $suiteName")
        description?.let { sb.appendLine(it) }
        sb.appendLine("Metrics: ${metrics.joinToString(", ")}")
        sb.appendLine()
        cases.forEach { case ->
            sb.appendLine("Case: ${case.name}")
            case.description?.let { sb.appendLine("  $it") }
            sb.appendLine("  Iterations: ${case.iterations} (warmup ${case.warmupIterations})")
            sb.appendLine("  Total duration: ${case.totalDuration}")
            sb.appendLine("  Per iteration: ${case.durationPerIteration}")
            case.latency?.let { latency ->
                sb.appendLine("  Latency samples: ${latency.count}")
                sb.appendLine("  Latency total: ${latency.totalDuration}")
                sb.appendLine("  Latency average: ${latency.averageDuration}")
                if (latency.totalsByOp.isNotEmpty()) {
                    sb.appendLine("  Latency by op:")
                    latency.totalsByOp.forEach { (op, duration) ->
                        sb.appendLine("    - $op -> $duration")
                    }
                }
            }
            case.memory?.let { memory ->
                sb.appendLine("  Memory peak: ${memory.peakUsedBytes} bytes")
                sb.appendLine("  Memory last: ${memory.lastUsedBytes} bytes")
                memory.usagePercent?.let { pct ->
                    sb.appendLine("  Memory usage: ${pct}%")
                }
            }
            sb.appendLine()
        }
        return sb.toString()
    }
}

public object BenchmarkRunner {

    public fun runSuite(suite: BenchmarkSuite): BenchmarkSuiteResult {
        val caseResults = suite.cases.map { case ->
            runCase(suite, case)
        }
        return BenchmarkSuiteResult(
            suiteName = suite.name,
            description = suite.description,
            metrics = suite.metrics,
            cases = caseResults
        )
    }

    private fun runCase(
        suite: BenchmarkSuite,
        case: BenchmarkCase
    ): BenchmarkCaseResult {
        val context = suite.contextFactory()
        val scope = BenchmarkExecutionScope(context)

        val observers = mutableListOf<ExecutionObserver>()
        var latencyObserver: LatencyExecutionObserver? = null
        var memoryObserver: MemorySnapshotObserver? = null

        if (suite.metrics.contains(BenchmarkMetric.LATENCY)) {
            latencyObserver = LatencyExecutionObserver()
            observers += latencyObserver
        }
        if (suite.metrics.contains(BenchmarkMetric.MEMORY)) {
            memoryObserver = MemorySnapshotObserver()
            observers += memoryObserver
        }

        observers.forEach { context.registerObserver(it) }

        try {
            case.setup?.invoke(scope)

            repeat(case.warmupIterations) {
                case.execute(scope)
            }

            observers.forEach { observer ->
                if (observer is ResettableExecutionObserver) {
                    observer.reset()
                }
            }

            val totalDuration = measureTime {
                repeat(case.iterations) {
                    case.execute(scope)
                }
            }

            val latencySummary = latencyObserver?.toSummary()
            val memorySummary = memoryObserver?.toSummary()
            val perIteration = if (case.iterations > 0) {
                totalDuration / case.iterations
            } else ZERO

            return BenchmarkCaseResult(
                name = case.name,
                description = case.description,
                warmupIterations = case.warmupIterations,
                iterations = case.iterations,
                totalDuration = totalDuration,
                durationPerIteration = perIteration,
                latency = latencySummary,
                memory = memorySummary
            )
        } finally {
            try {
                case.teardown?.invoke(scope)
            } finally {
                observers.forEach { context.unregisterObserver(it) }
                scope.clearStorage()
            }
        }
    }

    private fun LatencyExecutionObserver.toSummary(): LatencySummary {
        val samples = results()
        val total = samples.fold(ZERO) { acc, sample -> acc + sample.duration }
        val avg = if (samples.isNotEmpty()) total / samples.size else ZERO
        val byOp = samples.groupBy { it.opName }
            .mapValues { (_, values) ->
                values.fold(ZERO) { acc, sample -> acc + sample.duration }
            }
        reset()
        return LatencySummary(
            measurements = samples,
            totalDuration = total,
            averageDuration = avg,
            count = samples.size,
            totalsByOp = byOp
        )
    }

    private fun MemorySnapshotObserver.toSummary(): MemorySummary {
        val samples = results()
        val peak = samples.maxOfOrNull { it.usedBytes } ?: 0L
        val last = samples.lastOrNull()?.usedBytes ?: 0L
        val usagePercent = samples.lastOrNull()?.usagePercentage
        reset()
        return MemorySummary(
            samples = samples,
            peakUsedBytes = peak,
            lastUsedBytes = last,
            usagePercent = usagePercent
        )
    }
}
