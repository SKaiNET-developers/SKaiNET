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
)
