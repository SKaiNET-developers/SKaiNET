package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.Requantizers
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-008 Part B on views: `materialize` is the lossless direction (packed or narrow to dense
 * FP32), `requantizeTo` is the visible lossy call and shares [Requantizers] with the loaders.
 */
class ViewRepackRequantizeTest {

    private val shape = Shape(2, 32)
    private val values = FloatArray(64) { i -> (i - 32) * 0.173f }

    private fun denseView(): TensorView = TensorView.dense(Storage.Heap.wrap(values.copyOf()), shape)

    /** A packed view holding [values] quantized to [encoding], as a loader would produce. */
    private fun packedView(encoding: TensorEncoding): TensorView =
        Requantizers.quantize(encoding, values, shape).view!!

    private fun quantizedBytes(encoding: TensorEncoding, floats: FloatArray = values): ByteArray =
        (Requantizers.quantize(encoding, floats, shape) as PackedBlockStorage).packedData

    private fun bytesOf(view: TensorView): ByteArray =
        ByteArray(view.storage.sizeBytes.toInt()).also { view.storage.copyInto(it) }

    private fun narrowBytes(codec: NarrowFloatCodec): ByteArray {
        val bytes = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            bytes[i * 2] = (bits and 0xFF).toByte()
            bytes[i * 2 + 1] = ((bits shr 8) and 0xFF).toByte()
        }
        return bytes
    }

    // --- materialize: lossless ---

    @Test
    fun materialize_packedToDense_reproducesDequantizedValues_bitExact() {
        for (encoding in listOf(TensorEncoding.Q8_0, TensorEncoding.Q4_0, TensorEncoding.BITNET_PLANES)) {
            val source = packedView(encoding)
            val dequantized = source.toFloatArray()

            val dense = source.materialize()

            assertEquals(Format.dense(FP32), dense.format, encoding.name)
            assertEquals(shape, dense.shape, encoding.name)
            assertTrue(dense.storage.owner is Owner.Owned, encoding.name)
            assertContentEquals(dequantized, dense.toFloatArray(), encoding.name)
        }
    }

    @Test
    fun materialize_encodingsWithoutAQuantizer_reproduceDequantizedValues_bitExact() {
        // No quantizer produces these, so the blocks are synthetic: every byte below 0x3C keeps
        // the FP16 / FP32 scale fields finite. 512 elements is two K-quant super-blocks.
        val blockShape = Shape(2, 256)
        val encodings = listOf(
            TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
            TensorEncoding.BITNET_B1_58,
        )
        for (encoding in encodings) {
            val byteCount = checkNotNull(encoding.physicalBytes(blockShape.volume.toLong())).toInt()
            val bytes = ByteArray(byteCount) { ((it * 37 + 11) % 60).toByte() }
            val source = TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), blockShape, Format(FP32, encoding))
            val dequantized = source.toFloatArray()
            assertTrue(dequantized.any { it != 0f }, "$encoding: the synthetic blocks must decode to something")

            val dense = source.materialize()

            assertEquals(Format.dense(FP32), dense.format, "$encoding")
            assertEquals(blockShape, dense.shape, "$encoding")
            assertContentEquals(dequantized, dense.toFloatArray(), "$encoding")
        }
    }

    @Test
    fun materialize_narrowDense_widensExactly() {
        // Two bytes per element, not four: every narrow value is exactly representable in FP32.
        for ((dtype, codec) in listOf(FP16 to Fp16Codec, BF16 to Bf16Codec)) {
            val expected = FloatArray(values.size) { codec.decode(codec.encode(values[it])) }
            val narrow = TensorView.ofBytes(Storage.Heap.wrap(narrowBytes(codec), mutable = false), shape, Format.dense(dtype))
            assertEquals(values.size * 2L, narrow.storage.sizeBytes)

            val wide = narrow.materialize()

            assertEquals(Format.dense(FP32), wide.format, dtype.name)
            assertContentEquals(expected, wide.toFloatArray(), dtype.name)
        }
    }

    @Test
    fun narrowFloats_sameBytes_decodeDifferentlyByDtype() {
        val bytes = narrowBytes(Fp16Codec)
        val asFp16 = TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), shape, Format.dense(FP16)).toFloatArray()
        val asBf16 = TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), shape, Format.dense(BF16)).toFloatArray()
        assertTrue(asFp16.indices.any { asFp16[it] != asBf16[it] }, "the dtype, not the byte width, picks the codec")
    }

    @Test
    fun materialize_lossyDirection_throwsAndNamesRequantizeTo() {
        val failure = assertFailsWith<IllegalArgumentException> {
            denseView().materialize(Format(FP32, TensorEncoding.Q8_0))
        }
        assertTrue(
            failure.message.orEmpty().contains("requantizeTo"),
            "the error must point at requantizeTo, got: ${failure.message}"
        )
    }

    // --- requantizeTo: lossy, loader parity ---

    @Test
    fun requantizeTo_sameEncoding_returnsSameInstance() {
        val dense = denseView()
        assertSame(dense, dense.requantizeTo(TensorEncoding.Dense(4)))
        val q8 = packedView(TensorEncoding.Q8_0)
        assertSame(q8, q8.requantizeTo(TensorEncoding.Q8_0))
    }

    @Test
    fun requantizeTo_matchesLoaderQuantization_byteExact() {
        for (encoding in listOf(TensorEncoding.Q8_0, TensorEncoding.Q4_0, TensorEncoding.BITNET_PLANES)) {
            val result = denseView().requantizeTo(encoding)

            assertEquals(Format(FP32, encoding), result.format, encoding.name)
            assertEquals(shape, result.shape, encoding.name)
            assertContentEquals(quantizedBytes(encoding), bytesOf(result), encoding.name)
        }
    }

    @Test
    fun requantizeTo_q4_0_roundTripsThroughMaterialize() {
        val result = denseView().requantizeTo(TensorEncoding.Q4_0)
        // The lossless way back reproduces the quantizer's values.
        assertContentEquals(result.toFloatArray(), result.materialize().toFloatArray())
    }

    @Test
    fun requantizeTo_fromPackedSource_dequantizesFirst() {
        val q8 = packedView(TensorEncoding.Q8_0)
        val expected = quantizedBytes(TensorEncoding.Q4_0, q8.toFloatArray())

        val result = q8.requantizeTo(TensorEncoding.Q4_0)

        assertEquals(TensorEncoding.Q4_0, result.format.encoding)
        assertContentEquals(expected, bytesOf(result))
    }

    @Test
    fun requantizeTo_keepsTheTensorId() {
        val id = TensorId.parse("model.blk.0.attn.q_proj.weight")
        val named = TensorView.dense(Storage.Heap.wrap(values.copyOf()), shape, id = id)
        assertEquals(id, named.requantizeTo(TensorEncoding.Q8_0).id)
    }

    @Test
    fun requantizeTo_unsupportedEncoding_failsWithLoaderMessage() {
        val failure = assertFailsWith<IllegalArgumentException> {
            denseView().requantizeTo(TensorEncoding.Q4_K)
        }
        assertTrue(
            failure.message.orEmpty().contains("no quantizer for Q4_K"),
            "expected the loader's message, got: ${failure.message}"
        )
    }

    @Test
    fun requantizeTo_shapeThatDoesNotBlock_failsBeforeReadingTheStorage() {
        // Closed storage cannot be read, so this only passes if the shape validation runs first.
        val storage = Storage.Heap.floats(2 * 31)
        val unreadable = TensorView.dense(storage, Shape(2, 31))
        storage.close()
        val failure = assertFailsWith<IllegalArgumentException> {
            unreadable.requantizeTo(TensorEncoding.Q4_0)
        }
        assertTrue(
            failure.message.orEmpty().contains("multiple of 32"),
            "expected the Q4_0 blocking message, got: ${failure.message}"
        )
    }
}
