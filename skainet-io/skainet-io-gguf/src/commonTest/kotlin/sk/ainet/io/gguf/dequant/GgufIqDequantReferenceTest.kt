package sk.ainet.io.gguf.dequant

import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.GGMLQuantizationType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The llama.cpp reference for the IQ and MXFP4 block formats (#1409, parent #1408).
 *
 * Every type [DequantOps] decodes must reproduce llama.cpp's floats bit for bit. The types still
 * waiting for a decoder are listed in [PENDING]; the test fails as soon as one of them starts
 * decoding, so the lane that implements it has to take it off the list, and from then on it is
 * held to the reference like the others.
 */
class GgufIqDequantReferenceTest {

    /** Not implemented yet: #1410 (MXFP4), #1411 (IQ2), #1412 (IQ3), #1413 (IQ1). */
    private val PENDING: Set<GGMLQuantizationType> = setOf(
        GGMLQuantizationType.MXFP4,
        GGMLQuantizationType.IQ2_XXS, GGMLQuantizationType.IQ2_XS, GGMLQuantizationType.IQ2_S,
        GGMLQuantizationType.IQ3_XXS, GGMLQuantizationType.IQ3_S,
        GGMLQuantizationType.IQ1_S, GGMLQuantizationType.IQ1_M,
    )

    private fun hex(bits: Int): String = "0x" + bits.toUInt().toString(16).padStart(8, '0')

    @Test
    fun block_sizes_match_the_size_table() {
        for (ref in GgufIqDequantReference.ALL) {
            assertEquals(ref.blockElements to ref.blockBytes, GGML_QUANT_SIZES[ref.type], "GGML_QUANT_SIZES[${ref.type}]")
            for (case in ref.cases) {
                assertEquals(case.blocks * ref.blockBytes, case.bytes.size, "${ref.type}/${case.name}: raw bytes")
                assertEquals(case.blocks * ref.blockElements, case.expectedBits.size, "${ref.type}/${case.name}: expected values")
            }
        }
    }

    @Test
    fun reference_covers_the_eight_missing_types_and_the_two_iq4_types() {
        assertEquals(
            PENDING + setOf(GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.IQ4_XS),
            GgufIqDequantReference.ALL.map { it.type }.toSet(),
        )
    }

    @Test
    fun implemented_types_match_llama_cpp_bit_for_bit() {
        for (ref in GgufIqDequantReference.ALL.filter { it.type !in PENDING }) {
            for (case in ref.cases) {
                val want = case.expectedBits
                val got = DequantOps.dequantFromBytes(case.bytes, ref.type, want.size)
                assertEquals(want.size, got.size, "${ref.type}/${case.name}: element count")
                for (i in want.indices) {
                    if (got[i].toRawBits() != want[i]) {
                        fail(
                            "${ref.type}/${case.name} [$i] (block ${i / ref.blockElements}, element ${i % ref.blockElements}): " +
                                "got ${got[i]} (${hex(got[i].toRawBits())}), llama.cpp ${Float.fromBits(want[i])} (${hex(want[i])})"
                        )
                    }
                }
            }
        }
    }

    @Test
    fun pending_types_are_still_rejected() {
        for (ref in GgufIqDequantReference.ALL.filter { it.type in PENDING }) {
            val case = ref.cases.first()
            val e = assertFailsWith<IllegalStateException>("${ref.type} decodes now: remove it from PENDING so it is checked against the reference") {
                DequantOps.dequantFromBytes(case.bytes, ref.type, case.expectedBits.size)
            }
            assertTrue(e.message!!.contains("not implemented"), "${ref.type}: ${e.message}")
        }
    }

    @Test
    fun tables_are_the_ones_the_harness_printed() {
        val tables = mapOf(
            "KMASK_IQ2XS" to GgufIqTables.KMASK_IQ2XS,
            "KSIGNS_IQ2XS" to GgufIqTables.KSIGNS_IQ2XS,
            "IQ2XXS_GRID" to GgufIqTables.IQ2XXS_GRID,
            "IQ2XS_GRID" to GgufIqTables.IQ2XS_GRID,
            "IQ2S_GRID" to GgufIqTables.IQ2S_GRID,
            "IQ3XXS_GRID" to GgufIqTables.IQ3XXS_GRID,
            "IQ3S_GRID" to GgufIqTables.IQ3S_GRID,
            "IQ1S_GRID" to GgufIqTables.IQ1S_GRID,
            "KVALUES_MXFP4" to GgufIqTables.KVALUES_MXFP4,
            "KVALUES_IQ4NL" to GgufIqTables.KVALUES_IQ4NL,
        )
        assertEquals(GgufIqDequantReference.TABLE_CHECKSUMS.keys, tables.keys)
        for ((name, table) in tables) {
            val (size, checksum) = GgufIqDequantReference.TABLE_CHECKSUMS.getValue(name)
            assertEquals(size, table.size, "$name: size")
            var h = 0x811C9DC5L                                  // FNV-1a, 32 bit
            for (b in table) h = ((h xor (b.toLong() and 0xFF)) * 0x01000193L) and 0xFFFFFFFFL
            assertEquals(checksum, h, "$name: checksum")
        }
    }

    @Test
    fun table_spot_checks_against_ggml_common_h() {
        // iq2xxs_grid[1] = 0x080808080808082b: element 0 is the low byte.
        assertEquals(0x2b, GgufIqTables.IQ2XXS_GRID[8 * 1 + 0].toInt() and 0xFF)
        assertEquals(0x08, GgufIqTables.IQ2XXS_GRID[8 * 1 + 1].toInt() and 0xFF)
        // iq1s_grid[1] = 0xffffffffffffff01: +1 followed by seven -1.
        assertEquals(1, GgufIqTables.IQ1S_GRID[8 * 1 + 0].toInt())
        assertEquals(-1, GgufIqTables.IQ1S_GRID[8 * 1 + 1].toInt())
        // ksigns_iq2xs[1] = 129: bit 0 from the index, bit 7 for even parity.
        assertEquals(129, GgufIqTables.KSIGNS_IQ2XS[1].toInt() and 0xFF)
        assertEquals(listOf(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), GgufIqTables.KVALUES_MXFP4.map { it.toInt() })
        assertEquals(listOf(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113), GgufIqTables.KVALUES_IQ4NL.map { it.toInt() })
        assertEquals(0.125f, GgufIqTables.IQ1_DELTA)
    }
}
