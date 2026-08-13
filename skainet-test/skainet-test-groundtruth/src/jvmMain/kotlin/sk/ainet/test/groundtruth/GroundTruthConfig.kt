package sk.ainet.test.groundtruth

import java.io.File

/**
 * Configuration for ground truth validation.
 * Reads paths from system properties set by Gradle.
 */
public object GroundTruthConfig {

    /**
     * Directory containing generated GGUF ground truth files.
     * Set via Gradle system property 'groundtruth.results.dir'.
     */
    public val resultsDir: File by lazy {
        val path = System.getProperty("groundtruth.results.dir")
            ?: System.getenv("GROUNDTRUTH_RESULTS_DIR")
            ?: findDefaultResultsDir()
            ?: error(
                "Ground truth results directory not configured. " +
                "Run './gradlew generateGroundTruth' or set 'groundtruth.results.dir' system property."
            )
        File(path)
    }

    /**
     * Check if ground truth files are available.
     */
    public val isAvailable: Boolean
        get() = try {
            resultsDir.exists() && resultsDir.listFiles()?.isNotEmpty() == true
        } catch (e: Exception) {
            false
        }

    /**
     * When true, missing ground truth is a hard test failure instead of a skip — set via
     * `-PrequireGroundTruth=true` (see build.gradle.kts), used in CI so a broken/missing
     * pipeline shows up red instead of silently skipping every ground-truth test.
     * Local dev runs default to false: skip gracefully when the sibling
     * `../skainet-ground-truth` checkout or generated GGUF files aren't present.
     */
    public val requireAvailable: Boolean
        get() = System.getProperty("groundtruth.require")?.toBoolean() ?: false

    /**
     * Get all available test suites.
     */
    public val availableTestSuites: List<String>
        get() = if (isAvailable) {
            resultsDir.listFiles()
                ?.filter { it.isDirectory && it.name.startsWith("TS-") }
                ?.map { it.name }
                ?.sorted()
                ?: emptyList()
        } else {
            emptyList()
        }

    /**
     * Get the directory for a specific test suite.
     */
    public fun testSuiteDir(testSuite: String): File = File(resultsDir, testSuite)

    /**
     * Get all GGUF files for a test suite.
     */
    public fun ggufFiles(testSuite: String): List<File> {
        val dir = testSuiteDir(testSuite)
        return if (dir.exists()) {
            dir.listFiles()
                ?.filter { it.extension == "gguf" }
                ?.sortedBy { it.name }
                ?: emptyList()
        } else {
            emptyList()
        }
    }

    /**
     * Try to find the default results directory by walking up from the current directory.
     */
    private fun findDefaultResultsDir(): String? {
        var current = File(System.getProperty("user.dir"))
        repeat(5) {
            val candidate = current.resolve("../skainet-ground-truth/pytorch/results")
            if (candidate.exists()) {
                return candidate.canonicalPath
            }
            current = current.parentFile ?: return null
        }
        return null
    }
}

/**
 * Exception thrown when ground truth is not available.
 * Tests should catch this and skip gracefully.
 */
public class GroundTruthNotAvailableException(message: String) : RuntimeException(message)

/**
 * Check if ground truth is available, throw exception if not.
 * For use in test setup - tests should handle this exception.
 */
public fun requireGroundTruthAvailable() {
    if (!GroundTruthConfig.isAvailable) {
        throw GroundTruthNotAvailableException(
            "Ground truth not available. Run './gradlew generateGroundTruth' first."
        )
    }
}

/**
 * Check if a specific test suite is available, throw exception if not.
 */
public fun requireTestSuiteAvailable(testSuite: String) {
    requireGroundTruthAvailable()
    if (!GroundTruthConfig.testSuiteDir(testSuite).exists()) {
        throw GroundTruthNotAvailableException(
            "Test suite '$testSuite' not available in ground truth results."
        )
    }
}
