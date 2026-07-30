package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Pins [NarrowFloatInputMajorTensorData] — the relaid layout that makes transposing a KEEP_NATIVE
 * weight free (issue #888).
 *
 * The invariant under test throughout: input-major storage of `[rows, cols]` **is** row-major
 * storage of `[cols, rows]`, so [NarrowFloatInputMajorTensorData.transposedView] can share the
 * buffer outright. Element access has to stay correct on both sides of that view, which is what
 * distinguishes this from the K-quant lazy transpose (where `get()` on a transposed tensor is
 * meaningless and only the kernel's direct `packedData` read is valid).
 */
class NarrowFloatInputMajorTensorDataTest {

    /** 2x3, all exactly representable in both narrow formats so comparisons can be exact. */
    private val rows = 2
    private val cols = 3
    private val values = floatArrayOf(
        1.0f, 2.0f, 4.0f,
        8.0f, 0.5f, -2.0f,
    )

    private fun rowMajorBytes(codec: sk.ainet.lang.types.NarrowFloatCodec): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    @Test
    fun `element access matches the row-major original`() {
        val data = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(rows, cols), rowMajorBytes(Fp16Codec), Fp16Codec,
        )
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                assertEquals(
                    values[r * cols + c], data.get(r, c),
                    "get($r, $c) must see the logical element, not the relaid byte order",
                )
            }
        }
    }

    @Test
    fun `copyToFloatArray decodes in logical row-major order`() {
        val data = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(rows, cols), rowMajorBytes(Bf16Codec), Bf16Codec,
        )
        assertContentEquals(
            values, data.copyToFloatArray(),
            "consumers that do not care about storage must see the ordinary row-major sequence",
        )
    }

    @Test
    fun `transposedView shares the buffer and reads as the transpose`() {
        val data = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(rows, cols), rowMajorBytes(Fp16Codec), Fp16Codec,
        )
        val view = data.transposedView()

        assertSame(
            data.packedData, view.packedData,
            "the whole point is that no copy happens — a copy per forward is the bug being fixed",
        )
        assertEquals(Shape(cols, rows), view.shape)
        assertEquals(Fp16Codec, view.codec, "codec must survive; FP16 and BF16 are both 2 bytes")

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                assertEquals(
                    data.get(r, c), view.get(c, r),
                    "view[$c, $r] must be the transpose of data[$r, $c]",
                )
            }
        }
    }

    @Test
    fun `relayout round-trips back to the original bytes`() {
        // Relaying twice (with the shape flipped in between) is the identity — the cheapest
        // statement of "this is a transpose and not some other permutation".
        val original = rowMajorBytes(Fp16Codec)
        val once = NarrowFloatInputMajorTensorData.fromRowMajor(Shape(rows, cols), original, Fp16Codec)
        val twice = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(cols, rows), once.packedData, Fp16Codec,
        )
        assertContentEquals(original, twice.packedData)
    }

    @Test
    fun `a square weight is still genuinely transposed`() {
        // Square shapes are where an off-by-one in the relayout hides: the byte count matches
        // either way, so only the values reveal a wrong permutation.
        val square = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f)
        val bytes = ByteArray(square.size * 2)
        for (i in square.indices) {
            val bits = Fp16Codec.encode(square[i])
            bytes[i * 2] = (bits and 0xFF).toByte()
            bytes[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        val view = NarrowFloatInputMajorTensorData
            .fromRowMajor(Shape(2, 2), bytes, Fp16Codec)
            .transposedView()

        assertContentEquals(
            floatArrayOf(1.0f, 4.0f, 2.0f, 8.0f), view.copyToFloatArray(),
            "transposedView of [[1,2],[4,8]] must be [[1,4],[2,8]]",
        )
    }

    @Test
    fun `set writes through to the logical element`() {
        val data = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(rows, cols), rowMajorBytes(Fp16Codec), Fp16Codec,
        )
        data.set(1, 2, value = 16.0f)
        assertEquals(16.0f, data.get(1, 2))
        assertEquals(16.0f, data.transposedView().get(2, 1), "the shared buffer must see it too")
        assertEquals(values[0], data.get(0, 0), "neighbouring elements must be untouched")
    }

    @Test
    fun `rank other than two is rejected`() {
        // Norms are rank-1 and embeddings are gathered, not matmul'd; neither should ever be
        // relaid. Failing loudly beats silently mis-indexing them.
        assertFailsWith<IllegalArgumentException> {
            NarrowFloatInputMajorTensorData.fromRowMajor(Shape(4), ByteArray(8), Fp16Codec)
        }
        assertFailsWith<IllegalArgumentException> {
            NarrowFloatInputMajorTensorData(Shape(2, 2, 2), ByteArray(16), Fp16Codec)
        }
    }

    @Test
    fun `a short buffer is rejected`() {
        assertFailsWith<IllegalArgumentException> {
            NarrowFloatInputMajorTensorData.fromRowMajor(Shape(4, 4), ByteArray(8), Fp16Codec)
        }
    }

    @Test
    fun `the two codecs disagree on identical bytes`() {
        // Vacuity guard for every codec assertion above: both formats are 2 bytes per element,
        // so a codec mix-up cannot be caught by shape or size checks alone.
        val bytes = rowMajorBytes(Fp16Codec)
        val asFp16 = NarrowFloatInputMajorTensorData.fromRowMajor(Shape(rows, cols), bytes, Fp16Codec)
        val asBf16 = NarrowFloatInputMajorTensorData.fromRowMajor(Shape(rows, cols), bytes, Bf16Codec)
        assertTrue(
            !asFp16.copyToFloatArray().contentEquals(asBf16.copyToFloatArray()),
            "if these agreed, the codec-preservation assertions would prove nothing",
        )
    }
}
