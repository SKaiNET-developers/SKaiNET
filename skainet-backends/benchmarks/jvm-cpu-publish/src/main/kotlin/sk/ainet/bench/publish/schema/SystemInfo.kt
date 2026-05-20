package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class SystemInfo(
    val os: String,
    val arch: String,
    val cpu: String,
    @SerialName("cpu_logical_cores")
    val cpuLogicalCores: Int,
    @SerialName("memory_gib")
    val memoryGib: Long,
    val jdk: String,
    @SerialName("jdk_vendor")
    val jdkVendor: String,
    @SerialName("pts_client")
    val ptsClient: String? = null,
)
