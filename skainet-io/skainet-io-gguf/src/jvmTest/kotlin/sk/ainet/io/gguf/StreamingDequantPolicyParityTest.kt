package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Parity gate for the #782 fix: loading a GGUF with
 * [QuantPolicy.DEQUANTIZE_TO_FP32] (streaming per-tensor dequant into the
 * destination array, wrapped zero-copy) must produce bit-identical values to
 * the packed-block path that the loader has always used.
 *
 * Every supported quant format is exercised with *multiple blocks* of
 * pseudo-random codes — single-block tensors can pass by accident when block
 * indexing is broken.
 */
class StreamingDequantPolicyParityTest {

    @Test
    fun `dequantized load is bit-identical to packed load across all supported quant formats`() {
        val file = SyntheticGguf.write(
            SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
            SyntheticGguf.tensor("w_q5k", GGMLQuantizationType.Q5_K, elements = 1024),
            SyntheticGguf.tensor("w_q6k", GGMLQuantizationType.Q6_K, elements = 1024),
            SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 1024),
            SyntheticGguf.tensor("w_q40", GGMLQuantizationType.Q4_0, elements = 1024),
            SyntheticGguf.tensor("w_q50", GGMLQuantizationType.Q5_0, elements = 1024),
            SyntheticGguf.tensor("w_q51", GGMLQuantizationType.Q5_1, elements = 1024),
            SyntheticGguf.tensor("w_f16", GGMLQuantizationType.F16, elements = 1024),
            SyntheticGguf.tensor("w_bf16", GGMLQuantizationType.BF16, elements = 1024),
            SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        )
        try {
            val packedLoad = load(file, QuantPolicy.NATIVE_OPTIMIZED)
            val dequantLoad = load(file, QuantPolicy.DEQUANTIZE_TO_FP32)
            assertEquals(packedLoad.keys, dequantLoad.keys)

            for ((name, dequantTensor) in dequantLoad) {
                // Every tensor on the dequant path must be dense FP32 …
                val dense = dequantTensor.data
                assertTrue(
                    dense is FloatArrayTensorData<*>,
                    "$name: DEQUANTIZE_TO_FP32 must produce dense float storage, got ${dense::class.simpleName}",
                )
                val actual = dense.buffer

                val packedTensor = packedLoad.getValue(name)
                assertEquals(packedTensor.shape, dequantTensor.shape, "$name: shape parity")

                val packedData = packedTensor.data
                val expected: FloatArray = when (packedData) {
                    is PackedBlockStorage -> {
                        // … and match the packed block accessors bit-for-bit.
                        val out = FloatArray(packedData.shape.volume)
                        for (block in 0 until packedData.blockCount) {
                            packedData.dequantizeBlock(block, out, block * packedData.blockSize)
                        }
                        out
                    }
                    is FloatArrayTensorData<*> -> packedData.buffer
                    else -> error("$name: unexpected packed-path storage ${packedData::class.simpleName}")
                }

                assertEquals(expected.size, actual.size, "$name: element count")
                for (i in expected.indices) {
                    assertEquals(
                        expected[i].toRawBits(),
                        actual[i].toRawBits(),
                        "$name: value mismatch at flat index $i " +
                            "(packed=${expected[i]}, dequant=${actual[i]})",
                    )
                }
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `default policy is unchanged - quantized tensors stay packed`() {
        val file = SyntheticGguf.write(
            SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 512),
        )
        try {
            val loaded = load(file, QuantPolicy.NATIVE_OPTIMIZED)
            assertTrue(loaded.getValue("w_q4k").data is PackedBlockStorage)
        } finally {
            file.delete()
        }
    }

    @Test
    fun `RAW_BYTES policy is rejected eagerly`() {
        val e = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { error("must not be opened") },
                quantPolicy = QuantPolicy.RAW_BYTES,
            )
        }
        assertTrue("RAW_BYTES" in (e.message ?: ""))
    }

    private fun load(file: File, policy: QuantPolicy): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) },
                quantPolicy = policy,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }
}
