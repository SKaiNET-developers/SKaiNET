package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class BenchmarkRecord(
    @SerialName("schema_version")
    val schemaVersion: String,
    val suite: String,
    val scenario: String,
    @SerialName("published_at")
    val publishedAt: String,
    val runtime: RuntimeInfo,
    val system: SystemInfo,
    val config: RunConfig,
    val metrics: MetricSet,
    val samples: List<Double>,
    val unstable: Boolean = false,
    /**
     * Generation-loop metrics (#1035), present only for scenarios that run one — TTFT, tok/s,
     * effective bandwidth, page faults, per-module breakdown. `scripts/check_engine_json.sh`
     * validates the block whenever it appears.
     */
    val generation: GenerationMetricsRecord? = null,
)
