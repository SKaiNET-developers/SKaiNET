package sk.ainet.test.groundtruth

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.metrics.Accuracy
import sk.ainet.lang.nn.metrics.Metric
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * A metric case (SKEEP-009): batches in feed order and one expected value.
 *
 * In the fixture, `general.name` is `metric.<name>`, the inputs are pairs
 * (`input_0`, `input_1` = predictions and targets of batch 0, `input_2`, `input_3` = batch 1, …)
 * and `result` holds one element: the reference value over all batches together.
 */
public data class GroundTruthMetricCase(
    /** The metric's name, e.g. `"accuracy"` for `general.name = "metric.accuracy"`. */
    val metricName: String,
    val batches: List<Batch>,
    val expected: Double,
    /** The fixture's `op.*` values, strings included. */
    val params: Map<String, Any>,
    val source: GroundTruthTestCase,
) {
    public data class Batch(val predictions: GroundTruthTensor, val targets: GroundTruthTensor)
}

/** Prefix of `general.name` that marks a metric case. */
public const val METRIC_CASE_PREFIX: String = "metric."

/**
 * This case as a metric case, or `null` when it is not one. Fails when the inputs are not
 * `input_0 … input_n` in pairs, or when the result is not one element.
 */
public fun GroundTruthTestCase.toMetricCase(): GroundTruthMetricCase? {
    if (!operationName.startsWith(METRIC_CASE_PREFIX)) return null
    val where = sourcePath ?: description

    val ordered = inputs.entries
        .map { (name, tensor) ->
            val index = name.removePrefix("input_").toIntOrNull()
            require(name.startsWith("input_") && index != null) {
                "Metric case $where: input '$name' is not named input_<n>"
            }
            index to tensor
        }
        .sortedBy { it.first }
    require(ordered.isNotEmpty() && ordered.size % 2 == 0) {
        "Metric case $where: needs (predictions, targets) pairs, found ${ordered.size} input(s)"
    }
    ordered.forEachIndexed { position, (index, _) ->
        require(index == position) { "Metric case $where: input_$position is missing" }
    }
    require(expectedOutput.size == 1) {
        "Metric case $where: result must hold one value, found ${expectedOutput.size}"
    }

    return GroundTruthMetricCase(
        metricName = operationName.removePrefix(METRIC_CASE_PREFIX),
        batches = ordered.chunked(2) { (p, t) -> GroundTruthMetricCase.Batch(p.second, t.second) },
        expected = expectedOutput.data[0].toDouble(),
        params = rawOpParams,
        source = this,
    )
}

/**
 * Runs a [GroundTruthMetricCase] on a SKaiNET [Metric]: builds the metric from the case's
 * parameters, feeds every batch in order with `update()` and returns `compute()`.
 *
 * A metric name or a parameter that is not known fails the case with a message that names it.
 * A fixture that sets an option SKaiNET ignores would otherwise pass by luck.
 */
public class MetricExecutor(private val ctx: ExecutionContext) {

    public fun execute(case: GroundTruthMetricCase): Double {
        val metric = create(case)
        feed(metric, case)
        return metric.compute()
    }

    /** Builds the metric a case names, without feeding it. */
    public fun create(case: GroundTruthMetricCase): Metric {
        val factory = factories[case.metricName]
            ?: throw IllegalArgumentException(
                "Unknown metric '${case.metricName}'. Known: ${factories.keys.sorted()}"
            )
        val params = Params(case)
        params.targetEncoding() // required for every metric case
        val metric = factory(params)
        params.requireAllUsed()
        return metric
    }

    /** Feeds every batch of [case] to [metric], in order. */
    public fun feed(metric: Metric, case: GroundTruthMetricCase) {
        val encoding = Params(case).targetEncoding()
        for (batch in case.batches) {
            val predictions: Tensor<FP32, Float> =
                ctx.fromFloatArray(batch.predictions.shape, FP32::class, batch.predictions.data)
            val targets: Tensor<out DType, *> = when (encoding) {
                TargetEncoding.INDEX -> ctx.fromIntArray<Int32, Int>(
                    batch.targets.shape, Int32::class, toClassIndices(batch.targets, case)
                )
                TargetEncoding.ONE_HOT -> ctx.fromFloatArray<FP32, Float>(
                    batch.targets.shape, FP32::class, batch.targets.data
                )
            }
            metric.update(predictions, targets, ctx)
        }
    }

    private fun toClassIndices(targets: GroundTruthTensor, case: GroundTruthMetricCase): IntArray =
        IntArray(targets.size) { i ->
            val value = targets.data[i]
            val index = value.toInt()
            require(index.toFloat() == value && index >= 0) {
                "Metric case ${case.source.sourcePath ?: case.source.description}: " +
                    "target $value in '${targets.name}' is not a class index"
            }
            index
        }

    private enum class TargetEncoding { INDEX, ONE_HOT }

    /** The `op.*` values of one case; remembers which were read so the rest can be reported. */
    private class Params(private val case: GroundTruthMetricCase) {
        private val used = mutableSetOf<String>()

        fun int(name: String): Int? = read(name)?.let {
            (it as? Number)?.toInt() ?: fail(name, it, "an int")
        }

        fun float(name: String): Float? = read(name)?.let {
            (it as? Number)?.toFloat() ?: fail(name, it, "a float")
        }

        fun string(name: String): String? = read(name)?.let {
            it as? String ?: fail(name, it, "a string")
        }

        fun targetEncoding(): TargetEncoding = when (val value = string("target_encoding")) {
            "index" -> TargetEncoding.INDEX
            "one_hot" -> TargetEncoding.ONE_HOT
            null -> throw IllegalArgumentException(
                "Metric '${case.metricName}': op.target_encoding is required (index or one_hot)"
            )
            else -> throw IllegalArgumentException(
                "Metric '${case.metricName}': unknown op.target_encoding '$value' (index or one_hot)"
            )
        }

        fun requireAllUsed() {
            val unknown = case.params.keys - used
            require(unknown.isEmpty()) {
                "Metric '${case.metricName}': unknown parameter(s) ${unknown.sorted().map { "op.$it" }}"
            }
        }

        private fun read(name: String): Any? {
            used += name
            return case.params[name]
        }

        private fun fail(name: String, value: Any, expected: String): Nothing =
            throw IllegalArgumentException(
                "Metric '${case.metricName}': op.$name must be $expected, found '$value'"
            )
    }

    /** Metric name -> factory. Adding a metric to the harness is one entry here. */
    private val factories: Map<String, (Params) -> Metric> = mapOf(
        "accuracy" to { p -> Accuracy(dim = p.int("dim") ?: -1, threshold = p.float("threshold")) },
    )
}
