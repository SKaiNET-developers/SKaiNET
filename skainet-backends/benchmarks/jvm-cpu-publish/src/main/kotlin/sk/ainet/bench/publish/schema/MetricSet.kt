package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public data class MetricSet(
    @SerialName("primary_metric")
    val primaryMetric: String,
    val unit: String,
    @SerialName("value_mean")
    val valueMean: Double,
    @SerialName("value_stddev")
    val valueStddev: Double,
    @SerialName("value_min")
    val valueMin: Double,
    @SerialName("value_max")
    val valueMax: Double,
    @SerialName("cov_percent")
    val covPercent: Double,
)
