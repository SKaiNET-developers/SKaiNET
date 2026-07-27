package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.NarrowFloatDenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * Proves a KEEP_NATIVE weight survives `ctx.ops.matmul(x, ops.transpose(W))` still packed, so it
 * reaches the narrow matmul kernel instead of being widened elementwise (issue #888).
 *
 * Sibling of [PackedMatmulDispatchTest], and deliberately in `commonTest` for the same reason:
 * `DefaultCpuOps` and `DefaultCpuOpsJvm` intercept transpose separately, so only running on both
 * jvmTest and linuxX64Test proves both arms are wired.
 *
 * Weights are stored `[out, in]` as they are on disk, then transposed — the exact shape of what
 * `Linear.onForward` does on every forward pass.
 */
class NarrowFloatTransposeDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    private val outDim = 3
    private val inDim = 4

    /** `[out, in]`, all exactly representable in binary16 and bfloat16 alike. */
    private val weightRowMajor = floatArrayOf(
        1.0f, 2.0f, -1.0f, 0.5f,
        4.0f, -0.5f, 2.0f, 1.0f,
        -2.0f, 1.0f, 0.25f, 8.0f,
    )

    private val input = floatArrayOf(1.0f, -1.0f, 2.0f, 0.5f)

    private fun pack(values: FloatArray, codec: NarrowFloatCodec): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** `y[j] = sum_i x[i] * W[j][i]` — what `x.matmul(W.t())` must produce. */
    private fun reference(): FloatArray = FloatArray(outDim) { j ->
        var acc = 0.0f
        for (i in 0 until inDim) acc += input[i] * weightRowMajor[j * inDim + i]
        acc
    }

    private fun inputTensor(): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(1, inDim), FP32::class, input)

    @Suppress("UNCHECKED_CAST")
    private fun inputMajorWeight(codec: NarrowFloatCodec): Tensor<FP32, Float> {
        val data = NarrowFloatInputMajorTensorData.fromRowMajor(
            Shape(outDim, inDim), pack(weightRowMajor, codec), codec,
        )
        return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, label: String) {
        assertEquals(expected.size, actual.size, "$label: length")
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) < 1e-4f,
                "$label: element $i expected ${expected[i]} but was ${actual[i]}",
            )
        }
    }

    @Test
    fun `transpose of an input-major weight stays narrow and shares the buffer`() {
        for (codec in listOf(Fp16Codec, Bf16Codec)) {
            val w = inputMajorWeight(codec)
            val original = (w.data as NarrowFloatTensorData).packedData

            val wt = w.t()

            assertTrue(
                wt.data is NarrowFloatTensorData,
                "${codec.dtype.name}: transpose widened the weight — the kernel is unreachable",
            )
            assertTrue(
                wt.data is NarrowFloatDenseTensorData,
                "${codec.dtype.name}: expected a plain dense narrow tensor after transpose",
            )
            assertEquals(Shape(inDim, outDim), wt.shape, "${codec.dtype.name}: shape must swap")
            assertEquals(
                codec, (wt.data as NarrowFloatTensorData).codec,
                "${codec.dtype.name}: codec must survive the transpose",
            )
            assertSame(
                original, (wt.data as NarrowFloatTensorData).packedData,
                "${codec.dtype.name}: transpose must not copy — copying per forward is the bug",
            )
        }
    }

    @Test
    fun `matmul through the transpose matches the fp32 reference`() {
        val expected = reference()
        for (codec in listOf(Fp16Codec, Bf16Codec)) {
            val y = inputTensor().matmul(inputMajorWeight(codec).t())
            assertEquals(Shape(1, outDim), y.shape, "${codec.dtype.name}: output shape")
            assertClose(expected, y.data.copyToFloatArray(), codec.dtype.name)
        }
    }

    @Test
    fun `a row-major narrow weight is left alone by the lazy transpose`() {
        // The safety property. Only the input-major type may be reinterpreted; swapping the shape
        // of a row-major narrow buffer would silently produce a different matrix. The generic path
        // is slow, but it is correct — and correctness is what must not regress here.
        @Suppress("UNCHECKED_CAST")
        val rowMajor = ctx.fromData(
            NarrowFloatDenseTensorData(
                Shape(outDim, inDim), pack(weightRowMajor, Fp16Codec), Fp16Codec,
            ) as TensorData<FP32, Float>,
            FP32::class,
        )

        val wt = rowMajor.t()
        assertEquals(Shape(inDim, outDim), wt.shape)

        // Whatever representation the generic path chose, the values must be the true transpose.
        val got = wt.data.copyToFloatArray()
        for (j in 0 until outDim) {
            for (i in 0 until inDim) {
                assertEquals(
                    weightRowMajor[j * inDim + i], got[i * outDim + j],
                    "row-major transpose wrong at [$i, $j]",
                )
            }
        }
        assertClose(reference(), inputTensor().matmul(wt).data.copyToFloatArray(), "row-major")
    }

    @Test
    fun `the two codecs disagree on identical bytes`() {
        // Vacuity guard: both formats are 2 bytes per element, so nothing above would catch a
        // dispatch that picked the kernel by byte width instead of by codec.
        val bytes = pack(weightRowMajor, Fp16Codec)

        @Suppress("UNCHECKED_CAST")
        fun matmulAs(codec: NarrowFloatCodec): FloatArray {
            val data = NarrowFloatInputMajorTensorData.fromRowMajor(Shape(outDim, inDim), bytes, codec)
            val w = ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
            return inputTensor().matmul(w.t()).data.copyToFloatArray()
        }

        val asFp16 = matmulAs(Fp16Codec)
        val asBf16 = matmulAs(Bf16Codec)
        assertTrue(
            asFp16.indices.any { abs(asFp16[it] - asBf16[it]) > 1e-3f },
            "reading the same bytes under both codecs produced the same result — " +
                "the codec assertions in this class would prove nothing",
        )
    }
}
