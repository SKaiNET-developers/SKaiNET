package sk.ainet.test.groundtruth

import org.junit.Assume

/**
 * JUnit assumption helpers for ground truth tests.
 * These allow tests to be skipped when ground truth is not available.
 */

/**
 * Skip test if ground truth is not available.
 */
fun assumeGroundTruthAvailable() {
    Assume.assumeTrue(
        "Ground truth not available. Run './gradlew generateGroundTruth' first.",
        GroundTruthConfig.isAvailable
    )
}

/**
 * Skip test if a specific test suite is not available.
 */
fun assumeTestSuiteAvailable(testSuite: String) {
    assumeGroundTruthAvailable()
    Assume.assumeTrue(
        "Test suite '$testSuite' not available in ground truth results.",
        GroundTruthConfig.testSuiteDir(testSuite).exists()
    )
}
