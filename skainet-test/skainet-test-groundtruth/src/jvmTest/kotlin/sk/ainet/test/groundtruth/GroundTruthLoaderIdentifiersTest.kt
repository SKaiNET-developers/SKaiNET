package sk.ainet.test.groundtruth

import org.junit.Test
import kotlin.test.assertEquals

class GroundTruthLoaderIdentifiersTest {

    private fun ids(path: String?) = GroundTruthLoader.extractTestIdentifiers(path)

    @Test
    fun `the use case comes from the producer's file name`() {
        assertEquals(
            "TS-008" to "UC-003",
            ids("/work/results/TS-008/TS_008_UC_003.recall_macro_two_batches.gguf"),
        )
        assertEquals("TS-001" to "UC-001", ids("results/TS-001/TS_001_UC_001.basic_2D_convolution.gguf"))
    }

    @Test
    fun `the suite comes from the file name when no directory names it`() {
        assertEquals("TS-006" to "UC-004", ids("TS_006_UC_004.sub_channel_bias.gguf"))
    }

    @Test
    fun `a UC- path part still works`() {
        assertEquals("TS-002" to "UC-007", ids("results/TS-002/UC-007.gguf"))
    }

    @Test
    fun `a path without identifiers gives nulls`() {
        assertEquals(null to null, ids("/tmp/some_case.gguf"))
        assertEquals(null to null, ids(null))
    }
}
