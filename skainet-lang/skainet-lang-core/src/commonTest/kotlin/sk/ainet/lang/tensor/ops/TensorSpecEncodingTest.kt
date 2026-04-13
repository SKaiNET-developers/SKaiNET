package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class TensorSpecEncodingTest {

    @Test
    fun unset_encoding_reads_as_null() {
        val spec = TensorSpec(name = "x", shape = listOf(2, 3), dtype = "FP32")
        assertNull(spec.tensorEncoding)
    }

    @Test
    fun withTensorEncoding_round_trips_Q8_0() {
        val spec = TensorSpec(name = "w", shape = listOf(32), dtype = "FP32")
        val annotated = spec.withTensorEncoding(TensorEncoding.Q8_0)

        assertSame(TensorEncoding.Q8_0, annotated.tensorEncoding)
        // Original spec is untouched — TensorSpec is a data class and the
        // helper returns a copy.
        assertNull(spec.tensorEncoding)
    }

    @Test
    fun withTensorEncoding_round_trips_Q4_K() {
        val spec = TensorSpec(name = "w", shape = listOf(256), dtype = "FP32")
        val annotated = spec.withTensorEncoding(TensorEncoding.Q4_K)
        assertSame(TensorEncoding.Q4_K, annotated.tensorEncoding)
    }

    @Test
    fun withTensorEncoding_round_trips_TernaryPacked() {
        val spec = TensorSpec(name = "w", shape = listOf(128), dtype = "FP32")
        val annotated = spec.withTensorEncoding(TensorEncoding.TernaryPacked)
        assertSame(TensorEncoding.TernaryPacked, annotated.tensorEncoding)
    }

    @Test
    fun withTensorEncoding_round_trips_Dense() {
        val spec = TensorSpec(name = "x", shape = listOf(4), dtype = "FP32")
        val dense = TensorEncoding.Dense(bytesPerElement = 4)
        val annotated = spec.withTensorEncoding(dense)
        assertEquals(dense, annotated.tensorEncoding)
    }

    @Test
    fun passing_null_removes_the_encoding_entry() {
        val spec = TensorSpec(name = "w", shape = listOf(32), dtype = "FP32")
            .withTensorEncoding(TensorEncoding.Q8_0)
        assertSame(TensorEncoding.Q8_0, spec.tensorEncoding)

        val cleared = spec.withTensorEncoding(null)
        assertNull(cleared.tensorEncoding)
        assertTrue(
            !cleared.metadata.containsKey(TENSOR_ENCODING_METADATA_KEY),
            "clearing should remove the metadata key entirely, not leave a null"
        )
    }

    @Test
    fun withTensorEncoding_preserves_other_metadata() {
        val spec = TensorSpec(
            name = "w",
            shape = listOf(32),
            dtype = "FP32",
            metadata = mapOf("owner" to "attention.q_proj", "frozen" to true)
        )
        val annotated = spec.withTensorEncoding(TensorEncoding.Q8_0)

        assertEquals("attention.q_proj", annotated.metadata["owner"])
        assertEquals(true, annotated.metadata["frozen"])
        assertSame(TensorEncoding.Q8_0, annotated.tensorEncoding)
    }

    @Test
    fun replacing_encoding_overwrites_previous_value() {
        val spec = TensorSpec(name = "w", shape = listOf(32), dtype = "FP32")
            .withTensorEncoding(TensorEncoding.Q8_0)
            .withTensorEncoding(TensorEncoding.Q4_K)
        assertSame(TensorEncoding.Q4_K, spec.tensorEncoding)
    }
}
