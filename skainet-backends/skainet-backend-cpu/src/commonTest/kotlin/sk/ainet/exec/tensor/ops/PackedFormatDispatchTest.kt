package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.PackedLayoutFixtures
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.tensor.data.packedBlockTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * SKEEP-003 (#1507): the packed matmul is chosen from the weight's `Format` and layout, not from
 * its class. Every block encoding must give the same numbers whichever way the weight arrives —
 * canonical, stored in feed order, relayouted, or as a hand-built `[in, out]` label — and a packed
 * kernel, never the decoding reference, must serve it.
 */
class PackedFormatDispatchTest {
    private val ctx = DirectCpuExecutionContext()
    private val sink = RecordingTraceSink()
    private var previousSink = KernelDispatch.defaultSink

    @BeforeTest fun attachSink() { previousSink = KernelDispatch.defaultSink; KernelDispatch.defaultSink = sink }
    @AfterTest fun detachSink() { KernelDispatch.defaultSink = previousSink }

    private val encodings = listOf(
        TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0,
        TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
    )
    private val rows = PackedLayoutFixtures.ROWS
    private val blocksPerRow = PackedLayoutFixtures.BLOCKS_PER_ROW

    @Suppress("UNCHECKED_CAST")
    private fun weight(
        encoding: TensorEncoding,
        shape: Shape,
        bytes: ByteArray,
        order: BlockOrder = BlockOrder.ROW_MAJOR,
        blockAxis: Int = 1,
    ): Tensor<FP32, Float> =
        ctx.fromData(assertNotNull(packedBlockTensorData(encoding, shape, bytes, order, blockAxis)) as TensorData<FP32, Float>, FP32::class)

    private fun inDim(encoding: TensorEncoding) = blocksPerRow * encoding.blockSpec!!.blockSize

    private fun activation(vararg dims: Int): Tensor<FP32, Float> {
        val shape = Shape(*dims)
        return ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume) { ((it * 31) % 17 - 8) * 0.125f })
    }

    /** `x · Wᵀ` from the decoded weight, one accumulator per output. */
    private fun expected(x: FloatArray, w: FloatArray, k: Int): FloatArray {
        val m = x.size / k
        val out = FloatArray(m * rows)
        for (r in 0 until m) for (o in 0 until rows) {
            var acc = 0f
            for (i in 0 until k) acc += x[r * k + i] * w[o * k + i]
            out[r * rows + o] = acc
        }
        return out
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String) {
        assertEquals(expected.size, actual.size, "$what: size")
        for (i in expected.indices) {
            val tolerance = 1e-3f * maxOf(1f, abs(expected[i]))
            assertTrue(abs(expected[i] - actual[i]) <= tolerance, "$what: element $i expected ${expected[i]} got ${actual[i]}")
        }
    }

    private fun kernelsRun(): List<String> = sink.eventsOf<TraceEvent.KernelRun>().map { it.kernel }

    @Test
    fun everyArrivalOfAWeightGivesTheSameProduct() {
        for (encoding in encodings) {
            val name = encoding.name
            val k = inDim(encoding)
            val canonicalBytes = PackedLayoutFixtures.canonical(encoding)
            val feedBytes = PackedLayoutFixtures.kernelOrder(encoding)
            val canonical = weight(encoding, Shape(rows, k), canonicalBytes)
            val decoded = (canonical.data as PackedBlockStorage).toFloatArray()
            val x = activation(2, k)
            val want = expected(x.data.copyToFloatArray(), decoded, k)

            val viaPrimitive = ctx.ops.matmulWeightTransposed(x, canonical).data.copyToFloatArray()
            assertClose(want, viaPrimitive, "$name canonical")

            // Stored in feed order by a loader (#1120): the same shape, the order on the data.
            val stored = weight(encoding, Shape(rows, k), feedBytes, BlockOrder.INPUT_BLOCK_MAJOR)
            assertContentEquals(viaPrimitive, ctx.ops.matmulWeightTransposed(x, stored).data.copyToFloatArray(), "$name stored in feed order")

            // The relayout op: an [in, out] label that describes itself.
            val relayouted = ctx.ops.relayoutPackedWeightForKernels(canonical)
            assertEquals(Shape(k, rows), relayouted.shape, "$name relayout shape")
            val label = relayouted.data as PackedBlockStorage
            assertContentEquals(feedBytes, label.packedData, "$name relayout bytes")
            assertEquals(0, label.blockAxis, "$name relayout block axis")
            assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, label.blockOrder, "$name relayout order")
            assertContentEquals(viaPrimitive, ctx.ops.matmul(x, relayouted).data.copyToFloatArray(), "$name relayouted operand")

            // A hand-built label that does not describe itself is still read as feed order.
            val unmarked = weight(encoding, Shape(k, rows), feedBytes)
            assertContentEquals(viaPrimitive, ctx.ops.matmul(x, unmarked).data.copyToFloatArray(), "$name unmarked label")

            // x.matmul(w.t()) is the primitive.
            assertContentEquals(viaPrimitive, ctx.ops.matmul(x, ctx.ops.transpose(canonical)).data.copyToFloatArray(), "$name transpose marker")
        }
        val kernels = kernelsRun()
        assertTrue(kernels.isNotEmpty(), "the kernels must be visible in the trace")
        assertTrue(kernels.none { it.contains("reference", ignoreCase = true) }, "no reference kernel may run: $kernels")
    }

    @Test
    fun theRelayoutedLabelDecodesTheTransposedWeight() {
        for (encoding in encodings) {
            val k = inDim(encoding)
            val canonical = weight(encoding, Shape(rows, k), PackedLayoutFixtures.canonical(encoding))
            val decoded = (canonical.data as PackedBlockStorage).toFloatArray()
            val label = ctx.ops.relayoutPackedWeightForKernels(canonical).data as PackedBlockStorage
            val transposed = label.toFloatArray()
            for (o in 0 until rows) for (i in 0 until k) {
                assertEquals(decoded[o * k + i], transposed[i * rows + o], "${encoding.name} element ($o, $i)")
            }
        }
    }

    @Test
    fun activationsOfAnyRankAreServed() {
        for (encoding in encodings) {
            val k = inDim(encoding)
            val canonical = weight(encoding, Shape(rows, k), PackedLayoutFixtures.canonical(encoding))
            val relayouted = ctx.ops.relayoutPackedWeightForKernels(canonical)
            val decoded = (canonical.data as PackedBlockStorage).toFloatArray()

            val vector = activation(k)
            val one = ctx.ops.matmulWeightTransposed(vector, canonical)
            assertEquals(Shape(rows), one.shape, "${encoding.name} rank-1 shape")
            assertClose(expected(vector.data.copyToFloatArray(), decoded, k), one.data.copyToFloatArray(), "${encoding.name} rank-1")
            assertContentEquals(one.data.copyToFloatArray(), ctx.ops.matmul(vector, relayouted).data.copyToFloatArray())

            val batched = activation(2, 3, k)
            val three = ctx.ops.matmulWeightTransposed(batched, canonical)
            assertEquals(Shape(2, 3, rows), three.shape, "${encoding.name} rank-3 shape")
            assertClose(expected(batched.data.copyToFloatArray(), decoded, k), three.data.copyToFloatArray(), "${encoding.name} rank-3")
            assertContentEquals(three.data.copyToFloatArray(), ctx.ops.matmul(batched, relayouted).data.copyToFloatArray())
        }
        assertTrue(kernelsRun().none { it.contains("reference", ignoreCase = true) }, "no reference kernel may run: ${kernelsRun()}")
    }

    @Test
    fun aSlicedActivationIsCopiedNotMisread() {
        val encoding = TensorEncoding.Q8_0
        val k = inDim(encoding)
        val canonical = weight(encoding, Shape(rows, k), PackedLayoutFixtures.canonical(encoding))
        val decoded = (canonical.data as PackedBlockStorage).toFloatArray()
        val source = activation(4, k)
        val sliced = source.sliceView {              // rows 0 and 2: strided, not contiguous
            segment { step(0, 4, 2) }
            segment { range(0, k) }
        }
        val got = ctx.ops.matmulWeightTransposed(sliced, canonical).data.copyToFloatArray()
        assertClose(expected(sliced.data.copyToFloatArray(), decoded, k), got, "sliced activation")
    }
}
