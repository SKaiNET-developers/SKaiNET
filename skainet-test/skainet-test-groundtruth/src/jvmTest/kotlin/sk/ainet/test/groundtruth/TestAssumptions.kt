package sk.ainet.test.groundtruth

import org.junit.Assume

/**
 * JUnit assumption helpers for ground truth tests.
 *
 * By default these skip the test when ground truth isn't available — a normal state for
 * local dev, since generating it needs Docker + a sibling `../skainet-ground-truth`
 * checkout most contributors won't have. With [GroundTruthConfig.requireAvailable] set
 * (`-PrequireGroundTruth=true`, used in CI), the same conditions fail the test instead —
 * so a broken or unwired pipeline shows up red, not as an invisible skip.
 */

/**
 * Skip (or, if required, fail) the test if ground truth is not available.
 */
fun assumeGroundTruthAvailable() {
    if (GroundTruthConfig.requireAvailable) {
        check(GroundTruthConfig.isAvailable) {
            "Ground truth required (-PrequireGroundTruth=true) but not available. " +
                "Run './gradlew buildGroundTruthDocker generateGroundTruth' first."
        }
        return
    }
    Assume.assumeTrue(
        "Ground truth not available. Run './gradlew generateGroundTruth' first.",
        GroundTruthConfig.isAvailable
    )
}

/**
 * Skip (or, if required, fail) the test if a specific test suite is not available.
 */
fun assumeTestSuiteAvailable(testSuite: String) {
    assumeGroundTruthAvailable()
    if (GroundTruthConfig.requireAvailable) {
        check(GroundTruthConfig.testSuiteDir(testSuite).exists()) {
            "Test suite '$testSuite' required (-PrequireGroundTruth=true) but not available " +
                "in ground truth results."
        }
        return
    }
    Assume.assumeTrue(
        "Test suite '$testSuite' not available in ground truth results.",
        GroundTruthConfig.testSuiteDir(testSuite).exists()
    )
}
