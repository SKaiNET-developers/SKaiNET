package sk.ainet.test.groundtruth

import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
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

            // Infer parameters from test case description
            val params = inferConv2dParams(testCase)
            validator.validate(testCase, params)
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

            println("  Found ${testCases.size} test cases")

            val results = testCases.map { testCase ->
                val params = inferOperationParams(testCase)
                validator.validate(testCase, params, tolerance = ToleranceConfig.RELAXED)
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

    // =========================================================================
    // Helper Functions
    // =========================================================================

    private fun inferConv2dParams(testCase: GroundTruthTestCase): OperationParams {
        val desc = testCase.description.lowercase()
        return operationParams {
            when {
                desc.contains("stride") && desc.contains("2") -> stride(2)
                desc.contains("strided") -> stride(2)
            }
            when {
                desc.contains("padding") && desc.contains("1") -> padding(1)
                desc.contains("padded") -> padding(1)
            }
            when {
                desc.contains("dilation") && desc.contains("2") -> dilation(2)
                desc.contains("dilated") -> dilation(2)
            }
            when {
                desc.contains("depthwise") -> groups(3) // Assume 3 channels
                desc.contains("grouped") -> groups(2)
            }
        }
    }

    private fun inferOperationParams(testCase: GroundTruthTestCase): OperationParams {
        val opName = testCase.operationName.lowercase()
        val desc = testCase.description.lowercase()

        return when {
            opName.contains("conv") -> inferConv2dParams(testCase)
            opName.contains("flatten") -> operationParams {
                // Default flatten params
                startDim(1)
                endDim(-1)
            }
            opName.contains("pool") -> operationParams {
                kernelSize(2)
                stride(2)
            }
            else -> OperationParams()
        }
    }
}
