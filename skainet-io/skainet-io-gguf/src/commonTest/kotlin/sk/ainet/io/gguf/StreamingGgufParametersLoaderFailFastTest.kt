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
    fun q4_1_fails_the_pre_scan_with_tensor_name_and_supported_set() {
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(
                listOf(
                    tensorInfo("good", GGMLQuantizationType.Q8_0),
                    tensorInfo("blk.0.ffn_down.weight", GGMLQuantizationType.Q4_1),
                )
            )
        }
        val msg = e.message ?: ""
        assertTrue("blk.0.ffn_down.weight" in msg, "names the tensor: $msg")
        assertTrue("Q4_1" in msg, "names the type: $msg")
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
        val tensors = (0 until 12).map { tensorInfo("bad_$it", GGMLQuantizationType.Q4_1) }
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader.failFastOnUnsupportedTensorTypes(tensors)
        }
        val msg = e.message ?: ""
        assertTrue("12 tensor(s)" in msg, "reports the full count: $msg")
        assertTrue("and 4 more" in msg, "truncates the listing: $msg")
    }

    @Test
    fun quant_formats_with_load_branches_are_in_the_supported_set() {
        // Q4_0/Q5_0/Q5_1 gained load branches together with the fail-fast (#919);
        // this pins them so a refactor can't silently drop them back out.
        for (type in listOf(
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q8_0,
            GGMLQuantizationType.Q4_K,
        )) {
            assertTrue(
                type in StreamingGgufParametersLoader.SUPPORTED_TENSOR_TYPES,
                "$type should be supported",
            )
        }
    }
}
