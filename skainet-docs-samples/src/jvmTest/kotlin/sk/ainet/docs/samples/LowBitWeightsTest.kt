package sk.ainet.docs.samples

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Runs the sample behind `tutorials/low-bit-weights-getting-started.adoc` and checks the numbers
 * the page states, so the page cannot drift from the API or from what the code does.
 */
class LowBitWeightsTest {

    private fun run(): LowBitWeights.Result {
        val file = File.createTempFile("low-bit-weights", ".gguf")
        return try {
            LowBitWeights.run(file).also { println(LowBitWeights.render(it)) }
        } finally {
            file.delete()
        }
    }

    @Test
    fun oneViewTypeReadsEveryFormat() {
        val r = run()
        assertEquals(
            listOf("w.f32", "w.f16", "w.q8_0", "w.q4_0", "w.tq2_0", "w.tq1_0"),
            r.lines.map { it.name },
        )
        val bits = r.lines.associate { it.name to it.bitsPerWeight }
        assertEquals(32.0, bits.getValue("w.f32"))
        assertEquals(16.0, bits.getValue("w.f16"))
        assertEquals(8.5, bits.getValue("w.q8_0"))
        assertEquals(4.5, bits.getValue("w.q4_0"))
        assertEquals(2.0625, bits.getValue("w.tq2_0"))
        assertEquals(1.6875, bits.getValue("w.tq1_0"))

        // every packed form is logically FP32: the encoding says how it is stored
        for (l in r.lines.filter { it.name != "w.f16" }) {
            assertTrue(l.format.startsWith("Float32/"), "${l.name}: ${l.format}")
            assertEquals(LowBitWeights.ROWS * LowBitWeights.COLS * 4L, l.logicalBytes)
        }

        // the error grows as the bits shrink
        val err = r.lines.associate { it.name to it.maxError }
        assertEquals(0f, err.getValue("w.f32"))
        assertTrue(err.getValue("w.f16") < err.getValue("w.q8_0"))
        assertTrue(err.getValue("w.q8_0") < err.getValue("w.q4_0"))
        assertTrue(err.getValue("w.q4_0") < err.getValue("w.tq2_0"))
    }

    @Test
    fun theTrackerReportsLogicalAgainstPhysicalBytes() {
        val r = run()
        assertEquals(r.lines.sumOf { it.physicalBytes }, r.totalPhysicalBytes)
        assertTrue(r.totalPhysicalBytes < r.totalLogicalBytes)
    }

    @Test
    fun theMappedViewReadsTheSameValuesWithoutAHeapCopy() {
        val r = run()
        assertEquals("MMAP_FILE", r.mappedDomain)
        assertTrue(r.mappedEqualsHeap)
    }

    @Test
    fun materializeIsLosslessAndRequantizeIsNot() {
        val r = run()
        assertEquals("Float32/Dense(4B)", r.materializedFormat)
        assertTrue(r.materializeIsLossless)
        assertEquals("Float32/Q4_0", r.requantizedFormat)
        assertTrue(r.requantizeError > 0f)
    }

    @Test
    fun matmulRunsOnThePackedWeight() {
        val r = run()
        // a packed kernel, not the decoding reference
        assertTrue(r.kernel.endsWith("Q4_0"), "kernel: ${r.kernel}")
        for (i in r.reference.indices) assertEquals(r.reference[i], r.product[i], 1e-3f, "output $i")
    }
}
