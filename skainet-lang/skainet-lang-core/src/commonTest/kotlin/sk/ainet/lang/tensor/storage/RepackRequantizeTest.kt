package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Requantizers
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-008 Part B (#1433): `repackTo` re-encodes losslessly (packed/narrow -> Dense(4)),
 * `requantizeTo` is the visible lossy call, sharing [Requantizers] with the loaders.
 */
class RepackRequantizeTest {

    private val shape = Shape(2, 32)
    private val values = FloatArray(64) { i -> (i - 32) * 0.173f }

    private fun denseStorage(): TensorStorage = TensorStorageFactory.fromFloatArray(shape, values)

    /** Packed storage holding [values] quantized to [encoding], as a loader would produce. */
    private fun packedStorage(encoding: TensorEncoding): TensorStorage {
        val packed = Requantizers.quantize(encoding, values, shape) as PackedBlockStorage
        return TensorStorage(
            shape = shape,
            dtype = FP32,
            encoding = encoding,
            buffer = BufferHandleFactory.owned(packed.packedData),
        )
    }

    private fun floatsOf(storage: TensorStorage): FloatArray =
        TensorStorageFactory.toTensorData<FP32, Float>(storage).copyToFloatArray()

    // --- repackTo: lossless ---

    @Test
    fun repackTo_sameEncoding_returnsSameInstance() {
        val storage = denseStorage()
        assertSame(storage, storage.repackTo(TensorEncoding.Dense(4)))
    }

    @Test
    fun repackTo_q8_0_toDense_reproducesDequantizedValues_bitExact() {
        val source = packedStorage(TensorEncoding.Q8_0)
        val dequantized = floatsOf(source) // what dequantizeBlock gives

        val dense = source.repackTo(TensorEncoding.Dense(4))

        assertEquals(TensorEncoding.Dense(4), dense.encoding)
        assertEquals(FP32, dense.dtype)
        assertEquals(Ownership.OWNED, dense.ownership)
        assertEquals(shape, dense.shape)
        assertContentEquals(dequantized, floatsOf(dense))
    }

    @Test
    fun repackTo_q4_0_toDense_reproducesDequantizedValues_bitExact() {
        val source = packedStorage(TensorEncoding.Q4_0)
        val dequantized = floatsOf(source)

        val dense = source.repackTo(TensorEncoding.Dense(4))

        assertEquals(TensorEncoding.Dense(4), dense.encoding)
        assertContentEquals(dequantized, floatsOf(dense))
    }

    @Test
    fun repackTo_bitnetPlanes_toDense_reproducesDequantizedValues_bitExact() {
        val source = packedStorage(TensorEncoding.BITNET_PLANES)
        val dequantized = floatsOf(source)

        val dense = source.repackTo(TensorEncoding.Dense(4))

        assertEquals(TensorEncoding.Dense(4), dense.encoding)
        assertContentEquals(dequantized, floatsOf(dense))
    }

    @Test
    fun repackTo_narrowDense_widensExactly() {
        // Dense(2) FP16 bytes -> Dense(4) FP32; every FP16 value is exactly representable.
        val fp16Values = FloatArray(values.size) { Fp16Codec.decode(Fp16Codec.encode(values[it])) }
        val bytes = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            bytes[i * 2] = (bits and 0xFF).toByte()
            bytes[i * 2 + 1] = ((bits shr 8) and 0xFF).toByte()
        }
        val narrow = TensorStorageFactory.fromRawBytesOwned(shape, FP16, TensorEncoding.Dense(2), bytes)

        val wide = narrow.repackTo(TensorEncoding.Dense(4))

        assertEquals(TensorEncoding.Dense(4), wide.encoding)
        assertEquals(FP32, wide.dtype)
        assertContentEquals(fp16Values, floatsOf(wide))
    }

    @Test
    fun repackTo_lossyDirection_throwsAndNamesRequantizeTo() {
        val failure = assertFailsWith<UnsupportedOperationException> {
            denseStorage().repackTo(TensorEncoding.Q8_0)
        }
        assertTrue(
            failure.message.orEmpty().contains("requantizeTo"),
            "the error must point at requantizeTo, got: ${failure.message}"
        )
    }

    // --- requantizeTo: lossy, loader parity ---

    @Test
    fun requantizeTo_q8_0_matchesLoaderQuantization_byteExact() {
        val expected = (Requantizers.quantize(TensorEncoding.Q8_0, values, shape) as PackedBlockStorage).packedData

        val result = denseStorage().requantizeTo(TensorEncoding.Q8_0)

        assertEquals(TensorEncoding.Q8_0, result.encoding)
        assertEquals(FP32, result.dtype)
        assertEquals(Ownership.OWNED, result.ownership)
        assertEquals(shape, result.shape)
        val resultBytes = (result.buffer as BufferHandle.Owned).data
        assertContentEquals(expected, resultBytes)
    }

    @Test
    fun requantizeTo_q4_0_roundTripsThroughRepack() {
        val result = denseStorage().requantizeTo(TensorEncoding.Q4_0)
        val expected = (Requantizers.quantize(TensorEncoding.Q4_0, values, shape) as PackedBlockStorage).packedData

        assertContentEquals(expected, (result.buffer as BufferHandle.Owned).data)
        // And the lossless way back reproduces the quantizer's values.
        assertContentEquals(floatsOf(result), floatsOf(result.repackTo(TensorEncoding.Dense(4))))
    }

    @Test
    fun requantizeTo_bitnetPlanes_matchesLoaderQuantization() {
        val expected = (Requantizers.quantize(TensorEncoding.BITNET_PLANES, values, shape) as PackedBlockStorage).packedData

        val result = denseStorage().requantizeTo(TensorEncoding.BITNET_PLANES)

        assertEquals(TensorEncoding.BITNET_PLANES, result.encoding)
        assertContentEquals(expected, (result.buffer as BufferHandle.Owned).data)
    }

    @Test
    fun requantizeTo_fromPackedSource_dequantizesFirst() {
        val q8 = packedStorage(TensorEncoding.Q8_0)
        val expected = (Requantizers.quantize(TensorEncoding.Q4_0, floatsOf(q8), shape) as PackedBlockStorage).packedData

        val result = q8.requantizeTo(TensorEncoding.Q4_0)

        assertEquals(TensorEncoding.Q4_0, result.encoding)
        assertContentEquals(expected, (result.buffer as BufferHandle.Owned).data)
    }

    @Test
    fun requantizeTo_unsupportedEncoding_failsWithLoaderMessage() {
        val failure = assertFailsWith<IllegalArgumentException> {
            denseStorage().requantizeTo(TensorEncoding.Q4_K)
        }
        assertTrue(
            failure.message.orEmpty().contains("no quantizer for Q4_K"),
            "expected the loader's message, got: ${failure.message}"
        )
    }

    @Test
    fun requantizeTo_shapeThatDoesNotBlock_failsBeforeReadingTheBuffer() {
        // A device-resident buffer cannot be read on the host, so this only passes if the
        // shape validation runs first.
        val unreadable = TensorStorage(
            shape = Shape(2, 31),
            dtype = FP32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.DeviceResident(
                deviceId = "gpu0",
                backendHandle = Unit,
                sizeInBytes = 2L * 31 * 4,
                isMutable = false,
            ),
        )
        val failure = assertFailsWith<IllegalArgumentException> {
            unreadable.requantizeTo(TensorEncoding.Q4_0)
        }
        assertTrue(
            failure.message.orEmpty().contains("multiple of 32"),
            "expected the Q4_0 blocking message, got: ${failure.message}"
        )
    }
}
