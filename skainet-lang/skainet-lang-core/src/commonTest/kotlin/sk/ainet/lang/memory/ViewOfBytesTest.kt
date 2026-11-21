package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Requantizers
import sk.ainet.lang.tensor.data.packedBlockTensorData
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.tensor.storage.Ownership
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * `TensorView.ofBytes`: a view over a tensor's bytes as a model file has them, which is what the
 * streaming readers return. The same view must read a heap array and any other storage kind.
 */
class ViewOfBytesTest {

    private fun le32(values: IntArray): ByteArray {
        val out = ByteArray(values.size * 4)
        for (i in values.indices) for (b in 0 until 4) out[i * 4 + b] = (values[i] ushr (8 * b)).toByte()
        return out
    }

    private fun heap(bytes: ByteArray): Storage = Storage.Heap.wrap(bytes, mutable = false)

    /** The same bytes off the heap, where the target has such memory; `null` elsewhere. */
    private fun offHeap(bytes: ByteArray): Storage? {
        if (!PlatformStorage.supports(MemoryDomain.HOST_OFFHEAP)) return null
        val s = PlatformStorage.allocate(bytes.size.toLong(), MemoryDomain.HOST_OFFHEAP)
        s.copyFrom(bytes)
        return s
    }

    @Test
    fun denseFp32BytesAreReadLittleEndianWithoutACopy() {
        val values = floatArrayOf(1f, -2.5f, 3.25f, 0f, 0.125f, -7f)
        val bytes = le32(IntArray(values.size) { values[it].toRawBits() })
        val view = TensorView.ofBytes(heap(bytes), Shape(2, 3), Format.dense(FP32))

        assertSame(bytes, (view.storage as Storage.Heap).bytes, "the view is over the array it was given")
        assertFalse(view.isMutable)
        assertEquals(-2.5f, view.get(0, 1))
        assertEquals(0.125f, view.get(1, 1))
        assertContentEquals(values, view.toFloatArray())
    }

    @Test
    fun signedIntegersKeepTheirSign() {
        val i32 = TensorView.ofBytes(heap(le32(intArrayOf(-1, 2, Int.MIN_VALUE, 7))), Shape(4), Format.dense(Int32))
        assertContentEquals(floatArrayOf(-1f, 2f, Int.MIN_VALUE.toFloat(), 7f), i32.toFloatArray())

        val i8 = TensorView.ofBytes(heap(byteArrayOf(-128, -1, 0, 127)), Shape(4), Format.dense(Int8))
        assertContentEquals(floatArrayOf(-128f, -1f, 0f, 127f), i8.toFloatArray())
    }

    @Test
    fun narrowFloatsDecodeThroughTheirCodec() {
        val values = floatArrayOf(0.5f, -1f, 2f, 1024f)
        val bytes = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            bytes[i * 2] = bits.toByte()
            bytes[i * 2 + 1] = (bits ushr 8).toByte()
        }
        val view = TensorView.ofBytes(heap(bytes), Shape(4), Format.dense(FP16))
        assertEquals(TensorEncoding.Dense(2), view.format.encoding)
        assertContentEquals(values, view.toFloatArray())
    }

    @Test
    fun packedBlocksDecodeLikeTheDataClassOverTheSameBytes() {
        // Q8_0 and Q4_0 from the quantizers; Q4_K and Q6_K from a byte pattern with small f16 scales.
        val shape = Shape(2, 256)
        val floats = FloatArray(512) { ((it * 37) % 101 - 50) / 25f }
        val cases = ArrayList<Pair<TensorEncoding, ByteArray>>()
        for (enc in listOf(TensorEncoding.Q8_0, TensorEncoding.Q4_0)) {
            cases += enc to (Requantizers.quantize(enc, floats, shape) as PackedBlockStorage).packedData
        }
        for (enc in listOf(TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K, TensorEncoding.Q5_0, TensorEncoding.Q5_1)) {
            val n = enc.physicalBytes(512)!!.toInt()
            cases += enc to ByteArray(n) { ((it * 31 + 7) and 0x3F).toByte() }
        }
        for ((enc, bytes) in cases) {
            // The data class decodes block by block from its own array: the reference.
            val packed = packedBlockTensorData(enc, shape, bytes) as PackedBlockStorage
            val expected = FloatArray(512)
            for (b in 0 until packed.blockCount) packed.dequantizeBlock(b, expected, b * packed.blockSize)
            val view = TensorView.ofBytes(heap(bytes), shape, Format(FP32, enc))
            assertTrue(view.layout.blocked, enc.name)
            assertContentEquals(expected, view.toFloatArray(), "${enc.name} over heap bytes")
            assertEquals(expected[300], view.get(1, 44), "${enc.name} element read")

            val off = offHeap(bytes) ?: continue
            try {
                val offView = TensorView.ofBytes(off, shape, Format(FP32, enc))
                assertContentEquals(expected, offView.toFloatArray(), "${enc.name} over off-heap storage")
            } finally {
                off.close()
            }
        }
    }

    @Test
    fun ternaryBlocksDecodeFromHeapAndOffHeapStorage() {
        val values = FloatArray(512) { ((it % 3) - 1).toFloat() }
        for (enc in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0)) {
            val bytes = TernaryCodec.encode(enc, values)
            val expected = TernaryCodec.decode(enc, bytes, 512)
            val shape = Shape(2, 256)
            assertContentEquals(expected, TensorView.ofBytes(heap(bytes), shape, Format(FP32, enc)).toFloatArray(), enc.name)
            val off = offHeap(bytes) ?: continue
            try {
                assertContentEquals(expected, TensorView.ofBytes(off, shape, Format(FP32, enc)).toFloatArray(), "${enc.name} off-heap")
            } finally {
                off.close()
            }
        }
    }

    @Test
    fun aFormatNoDecoderReadsHasNoView() {
        val opaque = Format(FP32, TensorEncoding.Opaque("Q8_1", 40))
        val e = assertFailsWith<UnsupportedOperationException> { TensorView.ofBytes(heap(ByteArray(40)), Shape(32), opaque) }
        assertTrue(e.message!!.contains("Q8_1"), e.message)
        assertNull(StorageBlockDecoder.orNull(TensorEncoding.Dense(4)))
        assertNull(StorageBlockDecoder.orNull(TensorEncoding.BITNET_B1_58))
    }

    @Test
    fun storageTooSmallForTheShapeIsRejected() {
        assertFailsWith<IllegalArgumentException> { TensorView.ofBytes(heap(ByteArray(12)), Shape(4), Format.dense(FP32)) }
        assertFailsWith<IllegalArgumentException> { TensorView.ofBytes(heap(ByteArray(33)), Shape(32), Format(FP32, TensorEncoding.Q8_0)) }
    }

    @Test
    fun theTrackerRecordsWhatAViewSays() {
        val tracker = MemoryTracker()
        tracker.record("dense", TensorView.ofBytes(heap(ByteArray(16)), Shape(4), Format.dense(FP32)))
        tracker.record("packed", TensorView.ofBytes(heap(ByteArray(34)), Shape(32), Format(FP32, TensorEncoding.Q8_0)))
        tracker.record("owned", TensorView.dense(Storage.Heap.floats(8), Shape(8)))

        val report = tracker.report()
        assertEquals(3, report.tensorCount)
        assertEquals(2, report.borrowedCount)
        assertEquals(1, report.ownedCount)
        assertEquals(0, report.fileBackedCount)
        assertEquals(16L + 128L + 32L, report.totalLogicalBytes)
        assertEquals(16L + 34L + 32L, report.totalPhysicalBytes)
        assertEquals(Ownership.BORROWED, report.entries.first { it.name == "packed" }.report.ownership)
    }
}
