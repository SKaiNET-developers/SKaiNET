package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Unit tests for `StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes` —
 * the eager pre-scan that rejects GGUF files containing tensor types the loader
 * cannot materialize, instead of silently skipping them and shipping a model with
 * missing weights (#919).
 */
class StreamingGgufParametersLoaderFailFastTest {

    private fun tensorInfo(
        name: String,
        type: GGMLQuantizationType,
        rawTypeValue: Int = type.value,
    ): StreamingTensorInfo = StreamingTensorInfo(
        name = name,
        shape = listOf(32u),
        tensorType = type,
        rawTypeValue = rawTypeValue,
        nElements = 32,
        nBytes = 32,
        relativeOffset = 0,
        absoluteDataOffset = 0,
    )

    @Test
    fun supported_types_pass_the_pre_scan() {
        val tensors = StreamingGgufParametersLoader.SUPPORTED_TENSOR_TYPES.map {
            tensorInfo("t_${it.name}", it)
        }
        // No throw.
        StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(tensors)
    }

    @Test
    fun q8_k_fails_the_pre_scan_with_tensor_name_and_supported_set() {
        // Q8_K stays unsupported on purpose: it is a ggml dot-product intermediate that is
        // never written into GGUF files, so a file claiming it is malformed.
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(
                listOf(
                    tensorInfo("good", GGMLQuantizationType.Q8_0),
                    tensorInfo("blk.0.ffn_down.weight", GGMLQuantizationType.Q8_K),
                )
            )
        }
        val msg = e.message ?: ""
        assertTrue("blk.0.ffn_down.weight" in msg, "names the tensor: $msg")
        assertTrue("Q8_K" in msg, "names the type: $msg")
        assertTrue("Supported types" in msg, "lists the supported set: $msg")
        assertTrue("good" !in msg, "must not implicate supported tensors: $msg")
    }

    @Test
    fun unknown_raw_type_value_is_reported_verbatim() {
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(
                listOf(tensorInfo("weird", GGMLQuantizationType.UNKNOWN, rawTypeValue = 4711))
            )
        }
        val msg = e.message ?: ""
        assertTrue("4711" in msg, "reports the raw on-disk type value: $msg")
    }

    @Test
    fun long_offender_lists_are_truncated_with_a_count() {
        val tensors = (0 until 12).map { tensorInfo("bad_$it", GGMLQuantizationType.Q8_1) }
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(tensors)
        }
        val msg = e.message ?: ""
        assertTrue("12 tensor(s)" in msg, "reports the full count: $msg")
        assertTrue("and 4 more" in msg, "truncates the listing: $msg")
    }

    @Test
    fun quant_formats_with_load_branches_are_in_the_supported_set() {
        // Q4_0/Q5_0/Q5_1 gained load branches together with the fail-fast (#919), the
        // widen-to-FP32 types with the ggml-1.0 gate widening; this pins them so a
        // refactor can't silently drop them back out.
        for (type in listOf(
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.IQ4_NL,
            GGMLQuantizationType.IQ4_XS,
            GGMLQuantizationType.MXFP4,
            GGMLQuantizationType.IQ2_XXS,
            GGMLQuantizationType.IQ2_XS,
            GGMLQuantizationType.IQ2_S,
            GGMLQuantizationType.IQ3_XXS,
            GGMLQuantizationType.IQ3_S,
        )) {
            assertTrue(
                type in StreamingGgufParametersLoader.SUPPORTED_TENSOR_TYPES,
                "$type should be supported",
            )
        }
    }

    @Test
    fun ggml_intermediates_stay_rejected() {
        // Q8_1 and Q8_K have decoders, but ggml never writes them into GGUF files;
        // keeping them out of the gate keeps malformed files failing fast.
        for (type in listOf(GGMLQuantizationType.Q8_1, GGMLQuantizationType.Q8_K)) {
            assertTrue(
                type !in StreamingGgufParametersLoader.SUPPORTED_TENSOR_TYPES,
                "$type should stay unsupported",
            )
        }
    }
}
