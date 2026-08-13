package sk.ainet.test.groundtruth

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.FP32

/**
 * Validates SKaiNET operations against PyTorch ground truth.
 *
 * Usage:
 * ```kotlin
 * val validator = GroundTruthValidator(cpuOps)
 *
 * // Validate a test case
 * val result = validator.validate(testCase)
 * assertTrue(result.success)
 *
 * // Validate with custom tolerance
 * validator.validate(testCase, tolerance = 1e-4f)
 *
 * // Assert (throws on failure)
 * validator.assertValid(testCase)
 * ```
 */
public class GroundTruthValidator(
    private val ops: TensorOps,
    private val dataFactory: TensorDataFactory = DenseTensorDataFactory()
) {
    private val executor = OperationExecutor(ops, dataFactory)

    /**
     * Result of a validation run.
     */
    public data class ValidationResult(
        /** Whether the validation passed */
        val success: Boolean,

        /** The test case that was validated */
        val testCase: GroundTruthTestCase,

        /** Detailed comparison result for forward pass */
        val forwardComparison: ComparisonResult,

        /** Gradient comparison results (if gradients were validated) */
        val gradientComparisons: Map<String, ComparisonResult>? = null,

        /** Operation parameters used during validation */
        val params: OperationParams,

        /** Actual output tensor (for debugging) */
        val actualOutput: Tensor<FP32, Float>? = null,

        /** Error message if validation failed due to exception */
        val errorMessage: String? = null
    ) {
        /**
         * Generate a detailed report of the validation.
         */
        public fun toReport(): String {
            val sb = StringBuilder()
            sb.appendLine("=".repeat(60))
            sb.appendLine("Validation Report: ${testCase.description}")
            sb.appendLine("=".repeat(60))
            sb.appendLine("Operation: ${testCase.operationName}")
            sb.appendLine("Test Suite: ${testCase.testSuite ?: "N/A"}")
            sb.appendLine("Use Case: ${testCase.useCase ?: "N/A"}")
            sb.appendLine("Source: ${testCase.sourcePath ?: "N/A"}")
            sb.appendLine()

            if (errorMessage != null) {
                sb.appendLine("ERROR: $errorMessage")
                sb.appendLine()
            }

            sb.appendLine("Forward Pass: ${if (forwardComparison.success) "PASSED" else "FAILED"}")
            if (!forwardComparison.success) {
                sb.appendLine(forwardComparison.toErrorMessage())
            } else {
                sb.appendLine("  Max absolute difference: ${forwardComparison.maxAbsDiff}")
                sb.appendLine("  Max relative difference: ${forwardComparison.maxRelDiff}")
            }

            gradientComparisons?.let { grads ->
                sb.appendLine()
                sb.appendLine("Gradient Validation:")
                for ((inputName, comparison) in grads) {
                    val status = if (comparison.success) "PASSED" else "FAILED"
                    sb.appendLine("  $inputName: $status")
                    if (!comparison.success) {
                        sb.appendLine("    ${comparison.toErrorMessage()}")
                    }
                }
            }

            sb.appendLine()
            sb.appendLine("Overall: ${if (success) "PASSED" else "FAILED"}")
            sb.appendLine("=".repeat(60))

            return sb.toString()
        }
    }

    /**
     * Validate a ground truth test case.
     *
     * @param testCase The test case to validate
     * @param params Operation parameters (stride, padding, etc.) — defaults to
     *   [testCase]'s own `op.*` GGUF metadata ([GroundTruthTestCase.resolvedParams]),
     *   not an empty [OperationParams]; pass an explicit value to override.
     * @param tolerance Absolute tolerance for comparison (auto-selected if null)
     * @param rtol Relative tolerance for comparison
     * @param validateGradients Whether to also validate gradient computation
     */
    public fun validate(
        testCase: GroundTruthTestCase,
        params: OperationParams = testCase.resolvedParams(),
        tolerance: Float? = null,
        rtol: Float = 1e-5f,
        validateGradients: Boolean = false
    ): ValidationResult {
        val atol = tolerance ?: ToleranceConfig.forOperation(testCase.operationName)

        try {
            // Execute the operation in SKaiNET
            val actualOutput = executor.execute(testCase, params)

            // Compare forward pass output
            val forwardComparison = TensorAssertions.compare(
                expected = testCase.expectedOutput,
                actual = actualOutput,
                atol = atol,
                rtol = rtol
            )

            // Optionally validate gradients
            val gradientComparisons = if (validateGradients && testCase.expectedGradients != null) {
                validateGradientsInternal(testCase, actualOutput, atol, rtol)
            } else {
                null
            }

            val allGradientsPassed = gradientComparisons?.values?.all { it.success } ?: true
            val success = forwardComparison.success && allGradientsPassed

            return ValidationResult(
                success = success,
                testCase = testCase,
                forwardComparison = forwardComparison,
                gradientComparisons = gradientComparisons,
                params = params,
                actualOutput = actualOutput
            )

        } catch (e: Exception) {
            // Create a failed result with error message
            return ValidationResult(
                success = false,
                testCase = testCase,
                forwardComparison = ComparisonResult(
                    success = false,
                    maxAbsDiff = Float.MAX_VALUE,
                    maxRelDiff = Float.MAX_VALUE,
                    mismatchCount = testCase.expectedOutput.size,
                    totalElements = testCase.expectedOutput.size,
                    firstMismatchIndex = null,
                    firstMismatchExpected = null,
                    firstMismatchActual = null,
                    shapesMatch = false,
                    expectedShape = testCase.expectedOutput.shape,
                    actualShape = testCase.expectedOutput.shape
                ),
                params = params,
                errorMessage = "${e::class.simpleName}: ${e.message}"
            )
        }
    }

    /**
     * Validate multiple test cases.
     */
    public fun validateAll(
        testCases: List<GroundTruthTestCase>,
        paramsProvider: (GroundTruthTestCase) -> OperationParams = { it.resolvedParams() },
        tolerance: Float? = null,
        rtol: Float = 1e-5f
    ): List<ValidationResult> {
        return testCases.map { testCase ->
            validate(testCase, paramsProvider(testCase), tolerance, rtol)
        }
    }

    /**
     * Validate and assert - throws if validation fails.
     */
    public fun assertValid(
        testCase: GroundTruthTestCase,
        params: OperationParams = testCase.resolvedParams(),
        tolerance: Float? = null,
        rtol: Float = 1e-5f
    ) {
        val result = validate(testCase, params, tolerance, rtol)
        if (!result.success) {
            throw AssertionError(result.toReport())
        }
    }

    private fun validateGradientsInternal(
        testCase: GroundTruthTestCase,
        actualOutput: Tensor<FP32, Float>,
        atol: Float,
        rtol: Float
    ): Map<String, ComparisonResult> {
        // Gradient validation would require backward pass execution
        // This is a placeholder for when gradient computation is implemented
        val results = mutableMapOf<String, ComparisonResult>()

        testCase.expectedGradients?.forEach { (inputName, expectedGrad) ->
            // TODO: Implement actual gradient computation via tape
            results[inputName] = ComparisonResult(
                success = false,
                maxAbsDiff = Float.MAX_VALUE,
                maxRelDiff = Float.MAX_VALUE,
                mismatchCount = expectedGrad.size,
                totalElements = expectedGrad.size,
                firstMismatchIndex = null,
                firstMismatchExpected = null,
                firstMismatchActual = null,
                shapesMatch = true,
                expectedShape = expectedGrad.shape,
                actualShape = expectedGrad.shape
            )
        }

        return results
    }

    public companion object {
        /**
         * Create a validator summary report for multiple results.
         */
        public fun summaryReport(results: List<ValidationResult>): String {
            val passed = results.count { it.success }
            val failed = results.size - passed

            val sb = StringBuilder()
            sb.appendLine("=".repeat(60))
            sb.appendLine("Ground Truth Validation Summary")
            sb.appendLine("=".repeat(60))
            sb.appendLine("Total tests: ${results.size}")
            sb.appendLine("Passed: $passed")
            sb.appendLine("Failed: $failed")
            if (results.isNotEmpty()) {
                val passRate = passed.toFloat() / results.size * 100
                sb.appendLine("Pass rate: ${((passRate * 10).toInt() / 10.0)}%")
            }
            sb.appendLine()

            if (failed > 0) {
                sb.appendLine("Failed tests:")
                results.filter { !it.success }.forEach { result ->
                    sb.appendLine("  - ${result.testCase.description}")
                    sb.appendLine("    ${result.testCase.sourcePath ?: "unknown source"}")
                    result.errorMessage?.let { sb.appendLine("    Error: $it") }
                }
            }

            sb.appendLine("=".repeat(60))
            return sb.toString()
        }
    }
}

/**
 * DSL extension for fluent test case validation.
 */
public fun GroundTruthTestCase.validateWith(
    ops: TensorOps,
    params: OperationParams = resolvedParams(),
    tolerance: Float? = null
): GroundTruthValidator.ValidationResult {
    return GroundTruthValidator(ops).validate(this, params, tolerance)
}

/**
 * DSL extension for asserting test case validity.
 */
public fun GroundTruthTestCase.assertValidWith(
    ops: TensorOps,
    params: OperationParams = resolvedParams(),
    tolerance: Float? = null
) {
    GroundTruthValidator(ops).assertValid(this, params, tolerance)
}
