package sk.ainet.io.gguf.dequant

import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.GGMLQuantizationType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The llama.cpp reference for the legacy, K-quant, ternary and ggml-1.0 block formats —
 * the counterpart of [GgufIqDequantReferenceTest] for [GgufQuantDequantReference].
 *
 * Every type [DequantOps] decodes must reproduce llama.cpp's floats bit for bit. The types still
 * waiting for a decoder are listed in [PENDING]; the test fails as soon as one of them starts
 * decoding, so the lane that implements it has to take it off the list, and from then on it is
 * held to the reference like the others.
 *
 * Q8_1 has no reference: upstream has no `dequantize_row_q8_1` (a dot-product intermediate,
 * never written into GGUF files).
 */
class GgufQuantDequantReferenceTest {

    /** Not implemented yet: the remaining ggml-1.0 types. */
    private val PENDING: Set<GGMLQuantizationType> = setOf(
        GGMLQuantizationType.Q1_0, GGMLQuantizationType.Q2_0, GGMLQuantizationType.F8_E4M3,
    )

    private fun hex(bits: Int): String = "0x" + bits.toUInt().toString(16).padStart(8, '0')

    @Test
    fun block_sizes_match_the_size_table() {
        for (ref in GgufQuantDequantReference.ALL) {
            assertEquals(ref.blockElements to ref.blockBytes, GGML_QUANT_SIZES[ref.type], "GGML_QUANT_SIZES[${ref.type}]")
            for (case in ref.cases) {
                assertEquals(case.blocks * ref.blockBytes, case.bytes.size, "${ref.type}/${case.name}: raw bytes")
                assertEquals(case.blocks * ref.blockElements, case.expectedBits.size, "${ref.type}/${case.name}: expected values")
            }
        }
    }

    @Test
    fun reference_covers_the_legacy_k_ternary_and_ggml_1_0_types() {
        assertEquals(
            PENDING + setOf(
                GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1,
                GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1, GGMLQuantizationType.Q8_0,
                GGMLQuantizationType.Q2_K, GGMLQuantizationType.Q3_K, GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_K,
                GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0,
                GGMLQuantizationType.NVFP4,
            ),
            GgufQuantDequantReference.ALL.map { it.type }.toSet(),
        )
    }

    @Test
    fun implemented_types_match_llama_cpp_bit_for_bit() {
        for (ref in GgufQuantDequantReference.ALL.filter { it.type !in PENDING }) {
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
        for (ref in GgufQuantDequantReference.ALL.filter { it.type in PENDING }) {
            val case = ref.cases.first()
            val e = assertFailsWith<IllegalStateException>("${ref.type} decodes now: remove it from PENDING so it is checked against the reference") {
                DequantOps.dequantFromBytes(case.bytes, ref.type, case.expectedBits.size)
            }
            assertTrue(e.message!!.contains("not implemented"), "${ref.type}: ${e.message}")
        }
    }

    @Test
    fun list_entry_point_and_block_info_agree_with_the_byte_path() {
        for (ref in GgufQuantDequantReference.ALL.filter { it.type !in PENDING }) {
            assertEquals(ref.blockBytes to ref.blockElements, DequantOps.blockInfoFor(ref.type), "blockInfoFor(${ref.type})")
            val case = ref.cases.first()
            val fromList = DequantOps.dequantFromList(case.bytes.toList(), ref.type, case.expectedBits.size)
            assertEquals(case.expectedBits.toList(), fromList.map { it.toRawBits() }, "${ref.type}: dequantFromList")
        }
    }
}
