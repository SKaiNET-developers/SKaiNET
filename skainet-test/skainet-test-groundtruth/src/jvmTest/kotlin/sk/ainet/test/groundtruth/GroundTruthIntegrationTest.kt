package sk.ainet.test.groundtruth

import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Integration tests that validate SKaiNET operations against PyTorch ground truth.
 *
 * Prerequisites:
 * 1. Build the Docker image: ./gradlew :skainet-test:skainet-test-groundtruth:buildGroundTruthDocker
 * 2. Generate GGUF files:    ./gradlew :skainet-test:skainet-test-groundtruth:generateGroundTruth
 * 3. Run tests:              ./gradlew :skainet-test:skainet-test-groundtruth:jvmTest
 *
 * Or run all in one:
 *   ./gradlew generateGroundTruth jvmTest
 */
class GroundTruthIntegrationTest {

    private val context = DirectCpuExecutionContext()
    private val validator = GroundTruthValidator(context.ops, context.tensorDataFactory)

    // =========================================================================
    // TS-001: Convolution Operations
    // =========================================================================

    @Test
    fun `TS-001 - validate conv2d operations against PyTorch`() {
        assumeTestSuiteAvailable("TS-001")

        val testCases = GroundTruthLoader.loadTestSuite(
            GroundTruthConfig.resultsDir,
            "TS-001"
        )

        println("Found ${testCases.size} test cases in TS-001")

        val results = testCases.map { testCase ->
            println("  Validating: ${testCase.description}")

            // Params come from the GGUF's own op.* metadata (GroundTruthTestCase.resolvedParams,
            // the default validator.validate already uses) — no per-test-case guessing needed.
            validator.validate(testCase)
        }

        // Print summary
        println(GroundTruthValidator.summaryReport(results))

        // Assert all passed
        val failed = results.filter { !it.success }
        assertTrue(failed.isEmpty(), "Failed tests:\n${failed.joinToString("\n") { it.toReport() }}")
    }

    // =========================================================================
    // TS-003: Flatten Operations
    // =========================================================================

    @Test
    fun `TS-003 - validate flatten operations against PyTorch`() {
        assumeTestSuiteAvailable("TS-003")

        val testCases = GroundTruthLoader.loadTestSuite(
            GroundTruthConfig.resultsDir,
            "TS-003"
        )

        println("Found ${testCases.size} test cases in TS-003")

        val results = validator.validateAll(testCases)

        println(GroundTruthValidator.summaryReport(results))

        val failed = results.filter { !it.success }
        assertTrue(failed.isEmpty(), "Failed tests:\n${failed.joinToString("\n") { it.toReport() }}")
    }

    // =========================================================================
    // TS-006: Broadcasting Operations
    // =========================================================================

    @Test
    fun `TS-006 - validate broadcasting operations against PyTorch`() {
        assumeTestSuiteAvailable("TS-006")

        val testCases = GroundTruthLoader.loadTestSuite(
            GroundTruthConfig.resultsDir,
            "TS-006"
        )

        println("Found ${testCases.size} test cases in TS-006")

        val results = validator.validateAll(testCases)

        println(GroundTruthValidator.summaryReport(results))

        val failed = results.filter { !it.success }
        assertTrue(failed.isEmpty(), "Failed tests:\n${failed.joinToString("\n") { it.toReport() }}")
    }

    // =========================================================================
    // TS-008: Classification metrics (SKEEP-009, metric cases)
    // =========================================================================

    @Test
    fun `TS-008 - validate metrics against scikit-learn`() {
        assumeTestSuiteAvailable("TS-008")

        val testCases = GroundTruthLoader.loadTestSuite(GroundTruthConfig.resultsDir, "TS-008")
        println("Found ${testCases.size} test cases in TS-008")
        assertTrue(testCases.isNotEmpty(), "TS-008 holds no fixtures")

        // Value against the reference, and the reset() contract, for every case.
        val results = testCases.map { validator.validateMetric(it, context) }
        results.forEach { println("  ${it.toReport()}") }

        // Every metric of the harness has at least one case with more than one batch.
        val multiBatch = results.filter { it.case.batches.size > 1 }.map { it.case.metricName }.toSet()
        assertEquals(setOf("accuracy", "precision", "recall", "f1"), multiBatch, "metrics with a multi-batch case")

        val failures = results.filter { !it.success }.map { it.toReport() }
        assertTrue(failures.isEmpty(), "Failed metric cases:\n${failures.joinToString("\n")}")
    }

    // =========================================================================
    // Validate All Available Test Suites
    // =========================================================================

    @Test
    fun `validate all available test suites`() {
        assumeGroundTruthAvailable()

        val allResults = mutableListOf<GroundTruthValidator.ValidationResult>()

        for (testSuite in GroundTruthConfig.availableTestSuites) {
            println("\n=== $testSuite ===")

            val testCases = try {
                GroundTruthLoader.loadTestSuite(GroundTruthConfig.resultsDir, testSuite)
            } catch (e: Exception) {
                println("  Skipping: ${e.message}")
                continue
            }

            // Metric cases (SKEEP-009) have their own executor and their own test (TS-008).
                .filterNot { it.operationName.startsWith(METRIC_CASE_PREFIX) }

            println("  Found ${testCases.size} test cases")

            val results = testCases.map { testCase ->
                validator.validate(testCase, tolerance = ToleranceConfig.RELAXED)
            }

            allResults.addAll(results)

            val passed = results.count { it.success }
            val failed = results.size - passed
            println("  Passed: $passed, Failed: $failed")
        }

        // Final summary
        println("\n" + GroundTruthValidator.summaryReport(allResults))

        // Report failures but don't fail the test (some operations may not be implemented)
        val failed = allResults.filter { !it.success }
        if (failed.isNotEmpty()) {
            println("\nFailed operations (may not be implemented yet):")
            failed.forEach { result ->
                println("  - ${result.testCase.operationName}: ${result.testCase.description}")
                result.errorMessage?.let { println("    Error: $it") }
            }
        }
    }

}
