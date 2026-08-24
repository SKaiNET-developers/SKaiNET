package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.io.model.StagingPolicy
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.MmapFloatTensorData
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1037: the streaming-dequant path (#782) and the mapped path are two configurations of **one**
 * loader — `quantPolicy × staging` — not two code paths that can drift.
 *
 * Every combination must produce the same numbers; only *where the bytes live* differs. That is
 * the whole claim, so it is asserted directly: four loads of the same file, compared element by
 * element, plus the storage type each staging is supposed to produce.
 */
class StagingPolicyParityTest {

    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 1024),
        SyntheticGguf.tensor("w_f16", GGMLQuantizationType.F16, elements = 1024),
    )

    private fun load(f: File, quant: QuantPolicy, staging: StagingPolicy): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                quantPolicy = quant,
                staging = staging,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    private fun values(t: Tensor<FP32, Float>): FloatArray = t.data.copyToFloatArray()

    @Test
    fun `staging never changes the numbers, for either quant policy`() {
        val f = file()
        try {
            // The claim of #1037: staging decides *where the bytes live*, quantPolicy decides *what
            // the values are*. So HEAP and MAPPED must agree element for element under each policy.
            // (Across policies they legitimately differ: a packed tensor's own `get` returns codes,
            // which is what StreamingDequantPolicyParityTest covers.)
            for (quant in listOf(QuantPolicy.NATIVE_OPTIMIZED, QuantPolicy.DEQUANTIZE_TO_FP32)) {
                val heap = load(f, quant, StagingPolicy.HEAP)
                val mapped = load(f, quant, StagingPolicy.MAPPED)
                assertEquals(heap.keys, mapped.keys, "$quant: tensor set")
                assertTrue(heap.isNotEmpty())
                for ((name, tensor) in mapped) {
                    assertContentEquals(values(heap.getValue(name)), values(tensor), "$quant: values of $name")
                    assertEquals(heap.getValue(name).shape, tensor.shape, "$quant: shape of $name")
                }
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `mapped staging serves dense F32 tensors from file-backed pages`() {
        val f = file()
        try {
            val heap = load(f, QuantPolicy.NATIVE_OPTIMIZED, StagingPolicy.HEAP)
            val mapped = load(f, QuantPolicy.NATIVE_OPTIMIZED, StagingPolicy.MAPPED)

            assertTrue(
                heap.getValue("w_f32").data is FloatArrayTensorData<*>,
                "heap staging keeps F32 on the heap, got ${heap.getValue("w_f32").data::class.simpleName}",
            )
            assertTrue(
                mapped.getValue("w_f32").data is MmapFloatTensorData<*>,
                "mapped staging must not copy F32 onto the heap, got ${mapped.getValue("w_f32").data::class.simpleName}",
            )
            // packed tensors still arrive as packed block data: their kernels take arrays until #973
            assertEquals(
                heap.getValue("w_q4k").data::class.simpleName,
                mapped.getValue("w_q4k").data::class.simpleName,
                "packed staging is unchanged by the mapping",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `mapped staging falls back to the heap when the source is not a file`() {
        val f = file()
        try {
            // a source with no path — the documented fallback, not a failure
            val ctx = DefaultDataExecutionContext()
            val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
            runBlocking {
                StreamingGgufParametersLoader(
                    sourceProvider = { PathlessSource(JvmRandomAccessSource.open(f)) },
                    staging = StagingPolicy.MAPPED,
                ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
            }
            assertTrue(loaded.getValue("w_f32").data is FloatArrayTensorData<*>, "no path to map: stays on the heap")
            assertContentEquals(
                values(load(f, QuantPolicy.NATIVE_OPTIMIZED, StagingPolicy.HEAP).getValue("w_f32")),
                values(loaded.getValue("w_f32")),
            )
        } finally {
            f.delete()
        }
    }

    /** A source that reads fine but cannot say where its bytes live (a Blob, a socket, a test). */
    private class PathlessSource(private val delegate: sk.ainet.io.RandomAccessSource) : sk.ainet.io.RandomAccessSource {
        override val size: Long get() = delegate.size
        override val filePath: String? get() = null
        override fun readAt(position: Long, length: Int): ByteArray = delegate.readAt(position, length)
        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int =
            delegate.readAt(position, buffer, offset, length)
        override fun close(): Unit = delegate.close()
    }
}
