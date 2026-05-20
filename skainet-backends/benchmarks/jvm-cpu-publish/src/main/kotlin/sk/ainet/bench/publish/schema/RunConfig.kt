package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class RunConfig(
    @SerialName("warmup_runs")
    val warmupRuns: Int,
    @SerialName("measured_runs")
    val measuredRuns: Int,
    val seed: Long,
    val parameters: Map<String, String>,
    @SerialName("jvm_args")
    val jvmArgs: List<String>,
    @SerialName("smoke_mode")
    val smokeMode: Boolean,
)
