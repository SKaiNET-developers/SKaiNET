package sk.ainet.bench.publish.runner

import sk.ainet.bench.publish.env.RuntimeInfoProvider
import sk.ainet.bench.publish.env.SystemInfoProvider
import sk.ainet.bench.publish.schema.BenchmarkRecord
import sk.ainet.bench.publish.schema.MetricSet
import sk.ainet.bench.publish.schema.RunConfig
import java.time.Instant
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

public class BenchmarkRunner(
    private val warmupRuns: Int,
    private val measuredRuns: Int,
    private val seed: Long,
    private val smokeMode: Boolean,
    private val schemaVersion: String,
    private val covLimitPercent: Double = 3.0,
) {
    init {
        require(warmupRuns >= 0) { "warmupRuns must be >= 0" }
        require(measuredRuns >= 1) { "measuredRuns must be >= 1" }
    }

    public fun run(scenario: Scenario): BenchmarkRecord {
        scenario.setup()
        try {
            repeat(warmupRuns) { scenario.runOnce() }
            val samples = DoubleArray(measuredRuns) { scenario.runOnce() }
            val metrics = metrics(samples, scenario)
            val record = BenchmarkRecord(
                schemaVersion = schemaVersion,
                suite = scenario.suite,
                scenario = scenario.id,
                publishedAt = Instant.now().toString(),
                runtime = RuntimeInfoProvider.collect(scenario.kernelProvider),
                system = SystemInfoProvider.collect(),
                config = RunConfig(
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                    seed = seed,
                    parameters = scenario.parameters,
                    jvmArgs = RuntimeInfoProvider.jvmArgs(),
                    smokeMode = smokeMode,
                ),
                metrics = metrics,
                samples = samples.toList(),
                unstable = metrics.covPercent > covLimitPercent,
            )
            return record
        } finally {
            scenario.teardown()
        }
    }

    private fun metrics(samples: DoubleArray, scenario: Scenario): MetricSet {
        val mean = samples.average()
        val variance = if (samples.size < 2) 0.0 else samples.sumOf { v -> (v - mean) * (v - mean) } / (samples.size - 1)
        val stddev = sqrt(variance)
        var lo = Double.POSITIVE_INFINITY
        var hi = Double.NEGATIVE_INFINITY
        for (v in samples) {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        val cov = if (mean == 0.0) 0.0 else (stddev / mean) * 100.0
        return MetricSet(
            primaryMetric = scenario.primaryMetric,
            unit = scenario.unit,
            valueMean = mean,
            valueStddev = stddev,
            valueMin = lo,
            valueMax = hi,
            covPercent = cov,
        )
    }
}
