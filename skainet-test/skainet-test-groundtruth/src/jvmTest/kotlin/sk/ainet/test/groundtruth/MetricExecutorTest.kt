package sk.ainet.test.groundtruth

import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Metric cases built by hand, so the metric path of the harness (SKEEP-009) is tested
 * without Docker or generated fixtures.
 */
class MetricExecutorTest {

    private val executor = MetricExecutor(DirectCpuExecutionContext())

    private fun tensor(name: String, shape: Shape, vararg values: Float) =
        GroundTruthTensor(name, shape, values)

    /** Two samples, three classes; the predicted classes are the given ones. */
    private fun scores(name: String, vararg predicted: Int): GroundTruthTensor {
        val data = FloatArray(predicted.size * 3)
        predicted.forEachIndexed { row, cls -> data[row * 3 + cls] = 1f }
        return GroundTruthTensor(name, Shape(predicted.size, 3), data)
    }

    private fun case(
        name: String = "metric.accuracy",
        inputs: List<GroundTruthTensor>,
        expected: Float,
        params: Map<String, Any> = mapOf("target_encoding" to "index"),
    ) = GroundTruthTestCase(
        description = "hand-built",
        operationName = name,
        inputs = inputs.associateBy { it.name },
        expectedOutput = tensor("result", Shape(1), expected),
        rawOpParams = params,
    )

    @Test
    fun `batches are fed in input order and accumulate`() {
        // Batch 0: predicted [0, 1], targets [0, 2] -> 1 of 2. Batch 1: predicted [2], target [2] -> 1 of 1.
        val testCase = case(
            inputs = listOf(
                // Out of order on purpose: the map's order must not matter.
                scores("input_2", 2), tensor("input_3", Shape(1), 2f),
                scores("input_0", 0, 1), tensor("input_1", Shape(2), 0f, 2f),
            ),
            expected = 2f / 3f,
        )
        val metricCase = testCase.toMetricCase()!!

        assertEquals("accuracy", metricCase.metricName)
        assertEquals(2, metricCase.batches.size)
        assertEquals("input_0", metricCase.batches[0].predictions.name)
        assertEquals("input_3", metricCase.batches[1].targets.name)
        assertEquals(2.0 / 3.0, executor.execute(metricCase), 1e-12)
        assertEquals(metricCase.expected, executor.execute(metricCase), 1e-6)
    }

    @Test
    fun `one-hot targets are read as floats`() {
        val testCase = case(
            inputs = listOf(scores("input_0", 0, 1), scores("input_1", 0, 2)),
            expected = 0.5f,
            params = mapOf("target_encoding" to "one_hot"),
        )
        assertEquals(0.5, executor.execute(testCase.toMetricCase()!!), 1e-12)
    }

    @Test
    fun `a tensor operation is not a metric case`() {
        assertNull(case(name = "relu", inputs = listOf(scores("input_0", 0)), expected = 0f).toMetricCase())
    }

    @Test
    fun `an odd number of inputs fails`() {
        val testCase = case(inputs = listOf(scores("input_0", 0)), expected = 1f)
        val e = assertFailsWith<IllegalArgumentException> { testCase.toMetricCase() }
        assertTrue("pairs" in e.message!!, e.message)
    }

    @Test
    fun `a result with more than one value fails`() {
        val testCase = case(
            inputs = listOf(scores("input_0", 0), tensor("input_1", Shape(1), 0f)),
            expected = 1f,
        ).copy(expectedOutput = tensor("result", Shape(2), 1f, 1f))
        assertFailsWith<IllegalArgumentException> { testCase.toMetricCase() }
    }

    @Test
    fun `an unknown metric fails with its name`() {
        val testCase = case(
            name = "metric.auroc",
            inputs = listOf(scores("input_0", 0), tensor("input_1", Shape(1), 0f)),
            expected = 1f,
        )
        val e = assertFailsWith<IllegalArgumentException> { executor.execute(testCase.toMetricCase()!!) }
        assertTrue("auroc" in e.message!!, e.message)
    }

    @Test
    fun `an unknown parameter fails with its name`() {
        val testCase = case(
            inputs = listOf(scores("input_0", 0), tensor("input_1", Shape(1), 0f)),
            expected = 1f,
            params = mapOf("target_encoding" to "index", "top_k" to 2),
        )
        val e = assertFailsWith<IllegalArgumentException> { executor.execute(testCase.toMetricCase()!!) }
        assertTrue("op.top_k" in e.message!!, e.message)
    }

    @Test
    fun `a missing target encoding fails`() {
        val testCase = case(
            inputs = listOf(scores("input_0", 0), tensor("input_1", Shape(1), 0f)),
            expected = 1f,
            params = emptyMap(),
        )
        val e = assertFailsWith<IllegalArgumentException> { executor.execute(testCase.toMetricCase()!!) }
        assertTrue("target_encoding" in e.message!!, e.message)
    }

    @Test
    fun `a target that is not a class index fails`() {
        val testCase = case(
            inputs = listOf(scores("input_0", 0), tensor("input_1", Shape(1), 0.5f)),
            expected = 1f,
        )
        assertFailsWith<IllegalArgumentException> { executor.execute(testCase.toMetricCase()!!) }
    }
}
