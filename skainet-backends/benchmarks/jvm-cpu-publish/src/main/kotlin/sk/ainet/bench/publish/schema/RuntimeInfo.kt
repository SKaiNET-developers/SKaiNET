package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class RuntimeInfo(
    val name: String = "skainet-engine",
    val variant: String = "upstream",
    val version: String,
    val commit: String,
    val backend: String,
    @SerialName("kernel_provider")
    val kernelProvider: String,
    @SerialName("available_providers")
    val availableProviders: List<String>,
)
