package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.model.StagingPolicy
import sk.ainet.io.model.WeightOrientation
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1115: the loader takes one `WeightForm` where it took three flags, and the change is a change of
 * *spelling only*.
 *
 * That is the claim worth testing, because it is the one that can quietly be false. Every
 * combination of the three deprecated parameters is loaded twice — once through them, once through
 * the `WeightForm` they map to — and the two must agree on shapes and on every element. If the
 * mapping is wrong anywhere, some cell of that product disagrees.
 */
@Suppress("DEPRECATION")
@OptIn(ExperimentalMemoryApi::class)
class WeightFormLoaderParityTest {

    /** Mixed encodings, and a 2-D weight so the shape axis has something to reverse. */
    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 768)
            .copy(dims = listOf(256L, 3L)),
        SyntheticGguf.tensor("w_f16", GGMLQuantizationType.F16, elements = 1024),
    )

    private fun loadVia(f: File, build: (() -> JvmRandomAccessSource) -> StreamingGgufParametersLoader):
        Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            build { JvmRandomAccessSource.open(f) }
                .load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun `every combination of the three flags loads identically through the form it maps to`() {
        val f = file()
        try {
            for (quant in listOf(QuantPolicy.NATIVE_OPTIMIZED, QuantPolicy.DEQUANTIZE_TO_FP32)) {
                for (staging in listOf(StagingPolicy.HEAP, StagingPolicy.MAPPED)) {
                    for (orientation in listOf(WeightOrientation.AS_STORED, WeightOrientation.OUT_IN)) {
                        val viaFlags = loadVia(f) { src ->
                            StreamingGgufParametersLoader(
                                sourceProvider = src,
                                quantPolicy = quant,
                                staging = staging,
                                weightOrientation = orientation,
                            )
                        }
                        val viaForm = loadVia(f) { src ->
                            StreamingGgufParametersLoader(
                                sourceProvider = src,
                                weightForm = WeightForm(
                                    encoding = when (quant) {
                                        QuantPolicy.DEQUANTIZE_TO_FP32 -> EncodingRequest.DequantizeTo(FP32)
                                        else -> EncodingRequest.KeepAsStored
                                    },
                                    shape = when (orientation) {
                                        WeightOrientation.OUT_IN -> WeightShapeOrientation.OUT_IN
                                        else -> WeightShapeOrientation.AS_STORED
                                    },
                                    residency = when (staging) {
                                        StagingPolicy.MAPPED -> WeightResidency.MAPPED
                                        else -> WeightResidency.HEAP
                                    },
                                ),
                            )
                        }

                        val label = "$quant/$staging/$orientation"
                        assertEquals(viaFlags.keys, viaForm.keys, "$label: different tensors came out")
                        for ((name, flagsTensor) in viaFlags) {
                            val formTensor = viaForm.getValue(name)
                            assertEquals(flagsTensor.shape, formTensor.shape, "$label: $name shape")
                            assertContentEquals(
                                flagsTensor.data.copyToFloatArray(),
                                formTensor.data.copyToFloatArray(),
                                "$label: $name values",
                            )
                        }
                    }
                }
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the default loader is the default form`() {
        val f = file()
        try {
            val implicit = loadVia(f) { src -> StreamingGgufParametersLoader(sourceProvider = src) }
            val explicit = loadVia(f) { src ->
                StreamingGgufParametersLoader(sourceProvider = src, weightForm = WeightForm.AS_STORED_ON_HEAP)
            }
            for ((name, tensor) in implicit) {
                assertContentEquals(
                    tensor.data.copyToFloatArray(), explicit.getValue(name).data.copyToFloatArray(), name,
                )
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `setting both a form and a flag is refused rather than silently resolved`() {
        // One of the two would have to lose, and a caller who set a flag believes it is in effect.
        val failure = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                staging = StagingPolicy.MAPPED,
                weightForm = WeightForm.AS_STORED_ON_HEAP,
            )
        }
        assertTrue(failure.message!!.contains("pass only the form"), failure.message!!)
    }

    @Test
    fun `KERNEL_FEED is refused for the reason it is refused`() {
        // Not "unsupported": the bytes are the easy part. The refusal is because packed TensorData
        // reads packedData as canonical row-major, so feed-order bytes decode wrong silently (#1120).
        val failure = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(order = WeightByteOrder.KERNEL_FEED),
            )
        }
        val message = failure.message!!
        assertTrue(message.contains("#1120"), "it names where this is being fixed: $message")
        assertTrue(message.contains("canonical row-major"), "and why it cannot be faked: $message")
    }

    @Test
    fun `an encoding request this loader cannot honour is refused up front`() {
        val requantize = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(
                    encoding = EncodingRequest.RequantizeTo(sk.ainet.lang.tensor.storage.TensorEncoding.Q8_0),
                ),
            )
        }
        assertTrue(requantize.message!!.contains("quantizer"), requantize.message!!)

        val toFp16 = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(
                    encoding = EncodingRequest.DequantizeTo(sk.ainet.lang.types.FP16),
                ),
            )
        }
        assertTrue(toFp16.message!!.contains("FP32 only"), toFp16.message!!)
    }
}
