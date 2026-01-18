package sk.ainet.test.groundtruth

import org.junit.Test
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.exec.tensor.ops.DefaultCpuOpsBase
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for the ground truth validation framework.
 */
class GroundTruthValidationTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = TestCpuOps(dataFactory)
    private val validator = GroundTruthValidator(ops, dataFactory)

    // =========================================================================
    // Unit Tests for TensorAssertions
    // =========================================================================

    @Test
    fun `TensorAssertions - identical arrays should pass`() {
        val data = floatArrayOf(1f, 2f, 3f, 4f)
        val shape = Shape(intArrayOf(2, 2))

        val result = TensorAssertions.compare(
            expected = data,
            actual = data.copyOf(),
            expectedShape = shape,
            actualShape = shape
        )

        assertTrue(result.success)
        assertEquals(0, result.mismatchCount)
        assertEquals(0f, result.maxAbsDiff)
    }

    @Test
    fun `TensorAssertions - arrays within tolerance should pass`() {
        val expected = floatArrayOf(1f, 2f, 3f, 4f)
        val actual = floatArrayOf(1.000001f, 2.000001f, 3.000001f, 4.000001f)
        val shape = Shape(intArrayOf(2, 2))

        val result = TensorAssertions.compare(
            expected = expected,
            actual = actual,
            expectedShape = shape,
            actualShape = shape,
            atol = 1e-5f
        )

        assertTrue(result.success)
    }

    @Test
    fun `TensorAssertions - arrays outside tolerance should fail`() {
        val expected = floatArrayOf(1f, 2f, 3f, 4f)
        val actual = floatArrayOf(1.1f, 2f, 3f, 4f)
        val shape = Shape(intArrayOf(2, 2))

        val result = TensorAssertions.compare(
            expected = expected,
            actual = actual,
            expectedShape = shape,
            actualShape = shape,
            atol = 1e-5f
        )

        assertFalse(result.success)
        assertEquals(1, result.mismatchCount)
        assertEquals(0, result.firstMismatchIndex)
    }

    @Test
    fun `TensorAssertions - shape mismatch should fail`() {
        val expected = floatArrayOf(1f, 2f, 3f, 4f)
        val actual = floatArrayOf(1f, 2f, 3f, 4f)
        val expectedShape = Shape(intArrayOf(2, 2))
        val actualShape = Shape(intArrayOf(4))

        val result = TensorAssertions.compare(
            expected = expected,
            actual = actual,
            expectedShape = expectedShape,
            actualShape = actualShape
        )

        assertFalse(result.success)
        assertFalse(result.shapesMatch)
    }

    @Test
    fun `TensorAssertions - NaN values should match`() {
        val expected = floatArrayOf(1f, Float.NaN, 3f)
        val actual = floatArrayOf(1f, Float.NaN, 3f)
        val shape = Shape(intArrayOf(3))

        val result = TensorAssertions.compare(
            expected = expected,
            actual = actual,
            expectedShape = shape,
            actualShape = shape
        )

        assertTrue(result.success)
    }

    @Test
    fun `TensorAssertions - infinity values should match`() {
        val expected = floatArrayOf(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
        val actual = floatArrayOf(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
        val shape = Shape(intArrayOf(2))

        val result = TensorAssertions.compare(
            expected = expected,
            actual = actual,
            expectedShape = shape,
            actualShape = shape
        )

        assertTrue(result.success)
    }

    // =========================================================================
    // Unit Tests for GroundTruthTestCase
    // =========================================================================

    @Test
    fun `GroundTruthTestCase - creation and accessors`() {
        val input = GroundTruthTensor(
            name = "input_0",
            shape = Shape(intArrayOf(1, 3, 4, 4)),
            data = FloatArray(48) { it.toFloat() }
        )
        val output = GroundTruthTensor(
            name = "result",
            shape = Shape(intArrayOf(1, 3, 2, 2)),
            data = FloatArray(12) { it.toFloat() }
        )

        val testCase = GroundTruthTestCase(
            description = "Test convolution",
            operationName = "conv2d",
            inputs = mapOf("input_0" to input),
            expectedOutput = output,
            testSuite = "TS-001",
            useCase = "UC-001"
        )

        assertEquals("conv2d", testCase.operationName)
        assertEquals(1, testCase.inputs.size)
        assertEquals(48, testCase.inputs["input_0"]?.size)
        assertEquals(12, testCase.expectedOutput.size)
    }

    // =========================================================================
    // Unit Tests for OperationExecutor
    // =========================================================================

    @Test
    fun `OperationExecutor - add operation`() {
        val executor = OperationExecutor(ops, dataFactory)

        val a = GroundTruthTensor("a", Shape(intArrayOf(2, 2)), floatArrayOf(1f, 2f, 3f, 4f))
        val b = GroundTruthTensor("b", Shape(intArrayOf(2, 2)), floatArrayOf(5f, 6f, 7f, 8f))

        val testCase = GroundTruthTestCase(
            description = "Add test",
            operationName = "add",
            inputs = mapOf("a" to a, "b" to b),
            expectedOutput = GroundTruthTensor("result", Shape(intArrayOf(2, 2)), floatArrayOf(6f, 8f, 10f, 12f))
        )

        val result = executor.execute(testCase)
        val resultData = TensorAssertions.extractFloatData(result)

        assertTrue(resultData.contentEquals(floatArrayOf(6f, 8f, 10f, 12f)))
    }

    @Test
    fun `OperationExecutor - relu operation`() {
        val executor = OperationExecutor(ops, dataFactory)

        val input = GroundTruthTensor("input", Shape(intArrayOf(4)), floatArrayOf(-2f, -1f, 1f, 2f))

        val testCase = GroundTruthTestCase(
            description = "ReLU test",
            operationName = "relu",
            inputs = mapOf("input" to input),
            expectedOutput = GroundTruthTensor("result", Shape(intArrayOf(4)), floatArrayOf(0f, 0f, 1f, 2f))
        )

        val result = executor.execute(testCase)
        val resultData = TensorAssertions.extractFloatData(result)

        assertTrue(resultData.contentEquals(floatArrayOf(0f, 0f, 1f, 2f)))
    }

    // =========================================================================
    // Unit Tests for GroundTruthValidator
    // =========================================================================

    @Test
    fun `GroundTruthValidator - validates add operation correctly`() {
        val a = GroundTruthTensor("a", Shape(intArrayOf(2, 2)), floatArrayOf(1f, 2f, 3f, 4f))
        val b = GroundTruthTensor("b", Shape(intArrayOf(2, 2)), floatArrayOf(5f, 6f, 7f, 8f))
        val expected = GroundTruthTensor("result", Shape(intArrayOf(2, 2)), floatArrayOf(6f, 8f, 10f, 12f))

        val testCase = GroundTruthTestCase(
            description = "Add validation test",
            operationName = "add",
            inputs = mapOf("a" to a, "b" to b),
            expectedOutput = expected
        )

        val result = validator.validate(testCase)

        assertTrue(result.success, "Add operation should pass validation")
        assertTrue(result.forwardComparison.success)
    }

    @Test
    fun `GroundTruthValidator - detects mismatch correctly`() {
        val a = GroundTruthTensor("a", Shape(intArrayOf(2, 2)), floatArrayOf(1f, 2f, 3f, 4f))
        val b = GroundTruthTensor("b", Shape(intArrayOf(2, 2)), floatArrayOf(5f, 6f, 7f, 8f))
        // Intentionally wrong expected output
        val wrongExpected = GroundTruthTensor("result", Shape(intArrayOf(2, 2)), floatArrayOf(0f, 0f, 0f, 0f))

        val testCase = GroundTruthTestCase(
            description = "Should fail validation",
            operationName = "add",
            inputs = mapOf("a" to a, "b" to b),
            expectedOutput = wrongExpected
        )

        val result = validator.validate(testCase)

        assertFalse(result.success, "Validation should fail with wrong expected output")
        assertFalse(result.forwardComparison.success)
        assertEquals(4, result.forwardComparison.mismatchCount)
    }

    @Test
    fun `GroundTruthValidator - generates meaningful report`() {
        val a = GroundTruthTensor("input", Shape(intArrayOf(4)), floatArrayOf(1f, 2f, 3f, 4f))
        val expected = GroundTruthTensor("result", Shape(intArrayOf(4)), floatArrayOf(1f, 2f, 3f, 4f))

        val testCase = GroundTruthTestCase(
            description = "Report test",
            operationName = "relu",
            inputs = mapOf("input" to a),
            expectedOutput = expected,
            testSuite = "TS-001",
            useCase = "UC-001"
        )

        val result = validator.validate(testCase)
        val report = result.toReport()

        assertTrue(report.contains("Report test"))
        assertTrue(report.contains("relu"))
        assertTrue(report.contains("TS-001"))
        assertTrue(report.contains("PASSED") || report.contains("FAILED"))
    }

    @Test
    fun `GroundTruthValidator - summary report shows stats`() {
        val validTestCase = GroundTruthTestCase(
            description = "Valid test",
            operationName = "add",
            inputs = mapOf(
                "a" to GroundTruthTensor("a", Shape(intArrayOf(2)), floatArrayOf(1f, 2f)),
                "b" to GroundTruthTensor("b", Shape(intArrayOf(2)), floatArrayOf(3f, 4f))
            ),
            expectedOutput = GroundTruthTensor("result", Shape(intArrayOf(2)), floatArrayOf(4f, 6f))
        )

        val invalidTestCase = GroundTruthTestCase(
            description = "Invalid test",
            operationName = "add",
            inputs = mapOf(
                "a" to GroundTruthTensor("a", Shape(intArrayOf(2)), floatArrayOf(1f, 2f)),
                "b" to GroundTruthTensor("b", Shape(intArrayOf(2)), floatArrayOf(3f, 4f))
            ),
            expectedOutput = GroundTruthTensor("result", Shape(intArrayOf(2)), floatArrayOf(0f, 0f))
        )

        val results = listOf(
            validator.validate(validTestCase),
            validator.validate(invalidTestCase)
        )

        val summary = GroundTruthValidator.summaryReport(results)

        assertTrue(summary.contains("Total tests: 2"))
        assertTrue(summary.contains("Passed: 1"))
        assertTrue(summary.contains("Failed: 1"))
    }

    // =========================================================================
    // Test CPU Ops Implementation
    // =========================================================================

    /**
     * Minimal CPU ops implementation for testing.
     */
    private class TestCpuOps(dataFactory: DenseTensorDataFactory) : DefaultCpuOpsBase(dataFactory)
}

/**
 * Test configuration data class for parameterized tests.
 */
data class GroundTruthTestConfig(
    val testSuite: String,
    val operationName: String,
    val tolerance: Float = ToleranceConfig.STANDARD,
    val paramsProvider: (GroundTruthTestCase) -> OperationParams = { OperationParams() }
)

/**
 * Factory for creating test configurations.
 *
 * Usage with JUnit 5 @TestFactory:
 * ```kotlin
 * @TestFactory
 * fun groundTruthTests(): List<DynamicTest> {
 *     val validator = GroundTruthValidator(cpuOps)
 *     val testCases = GroundTruthLoader.loadFromDirectory(File("ground-truth/gguf"))
 *
 *     return testCases.map { testCase ->
 *         DynamicTest.dynamicTest(testCase.description) {
 *             validator.assertValid(testCase)
 *         }
 *     }
 * }
 * ```
 */
object GroundTruthTestFactory {

    fun createTestConfigs(): List<GroundTruthTestConfig> = listOf(
        // Convolution tests
        GroundTruthTestConfig(
            testSuite = "TS-001",
            operationName = "conv2d",
            tolerance = ToleranceConfig.STANDARD,
            paramsProvider = { testCase ->
                when {
                    testCase.description.contains("strided", ignoreCase = true) ->
                        operationParams { stride(2) }
                    testCase.description.contains("padded", ignoreCase = true) ->
                        operationParams { padding(1) }
                    testCase.description.contains("dilated", ignoreCase = true) ->
                        operationParams { dilation(2) }
                    testCase.description.contains("depthwise", ignoreCase = true) ->
                        operationParams { groups(3) }
                    else -> OperationParams()
                }
            }
        ),

        // Flatten tests
        GroundTruthTestConfig(
            testSuite = "TS-003",
            operationName = "flatten",
            tolerance = ToleranceConfig.STRICT
        ),

        // Broadcasting tests
        GroundTruthTestConfig(
            testSuite = "TS-006",
            operationName = "add",
            tolerance = ToleranceConfig.STRICT
        )
    )
}
