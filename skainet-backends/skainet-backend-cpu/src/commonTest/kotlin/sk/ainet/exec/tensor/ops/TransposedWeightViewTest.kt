package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TransposedWeightTensorData
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * #1108: `w.t()` works for a packed weight, and means the same product it means for a dense one.
 *
 * #973 made `transpose` refuse packed weights, because transposing block-quantized bytes needs
 * requantization and what the code did instead was an O(bytes) relabel that fed kernels the wrong
 * order. Refusing was right about the operation and wrong about the caller: it made the code an
 * author writes depend on how the weight happened to be stored, which is a property of the file
 * loaded and the device it runs on.
 *
 * So `transpose` now returns `Wᵀ` unmaterialized and `matmul` recognises it. These tests hold that
 * to two standards: the product must be **bit-identical** to the primitive it routes to, and the
 * marker must not be usable as packed data by anything that goes looking.
 *
 * Every fixture is **three blocks per row**. At one block per row the canonical and kernel-feed
 * orders coincide, so a layout bug passes such a test vacuously (#968).
 */
class TransposedWeightViewTest {

    private val ctx = DirectCpuExecutionContext()
    private val outDim = 4
    private val blocksPerRow = 3

    /** name → (elements per block, bytes per block, builder) */
    private val encodings: List<Triple<String, Pair<Int, Int>, (Shape, ByteArray) -> TensorData<FP32, Float>>> =
        listOf(
            Triple("Q4_0", 32 to 18) { s, b -> Q4_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q5_0", 32 to 22) { s, b -> Q5_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q5_1", 32 to 24) { s, b -> Q5_1BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q8_0", 32 to 34) { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q4_K", 256 to 144) { s, b -> Q4_KBlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q5_K", 256 to 176) { s, b -> Q5_KBlockTensorData(s, b) as TensorData<FP32, Float> },
            Triple("Q6_K", 256 to 210) { s, b -> Q6_KBlockTensorData(s, b) as TensorData<FP32, Float> },
        )

    /**
     * The encodings whose block layout is simple enough to hand-build a *valid* block: an fp16
     * scale (and for Q5_1 an fp16 min) at the front, codes after. The K-quants pack hierarchical
     * sub-block scales, and random bytes there dequantize to NaN — which says nothing about block
     * order, so those are covered by the routing and safety tests above and by the existing golden
     * parity suite, not by the hand-rolled reference below.
     */
    private val simpleEncodings: List<Triple<String, Pair<Int, Int>, (Shape, ByteArray) -> TensorData<FP32, Float>>> =
        encodings.filter { it.first in setOf("Q4_0", "Q5_0", "Q5_1", "Q8_0") }

    /** [weight] with each block's scale set to 1.0 and min to 0.0, so dequantization is finite. */
    private fun validWeight(
        name: String,
        blockElems: Int,
        bytesPerBlock: Int,
        build: (Shape, ByteArray) -> TensorData<FP32, Float>,
    ): Pair<Tensor<FP32, Float>, Int> {
        val inDim = blockElems * blocksPerRow
        val bytes = ByteArray(outDim * blocksPerRow * bytesPerBlock)
        var seed = 7
        for (i in bytes.indices) {
            seed = seed * 1103515245 + 12345
            bytes[i] = ((seed ushr 16) and 0xFF).toByte()
        }
        for (b in 0 until outDim * blocksPerRow) {
            val base = b * bytesPerBlock
            bytes[base] = 0x00; bytes[base + 1] = 0x3C          // fp16 1.0
            if (name == "Q5_1") { bytes[base + 2] = 0x00; bytes[base + 3] = 0x00 }   // fp16 min 0.0
        }
        return ctx.fromData(build(Shape(outDim, inDim), bytes), FP32::class) to inDim
    }

    /** A weight with content that differs per block, so a wrong block order cannot go unnoticed. */
    private fun weight(blockElems: Int, bytesPerBlock: Int, build: (Shape, ByteArray) -> TensorData<FP32, Float>):
        Pair<Tensor<FP32, Float>, Int> {
        val inDim = blockElems * blocksPerRow
        val bytes = ByteArray(outDim * blocksPerRow * bytesPerBlock)
        var seed = 7
        for (i in bytes.indices) {
            seed = seed * 1103515245 + 12345
            bytes[i] = ((seed ushr 16) and 0xFF).toByte()
        }
        return ctx.fromData(build(Shape(outDim, inDim), bytes), FP32::class) to inDim
    }

    private fun activation(inDim: Int): Tensor<FP32, Float> =
        ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, FloatArray(inDim) { (it % 11) * 0.0625f })

    @Test
    fun `matmul over a transposed packed weight equals the primitive it routes to`() {
        for ((name, geom, build) in encodings) {
            val (w, inDim) = weight(geom.first, geom.second, build)
            val x = activation(inDim)

            val viaTranspose = x.matmul(w.t()).data.copyToFloatArray()
            val viaPrimitive = x.matmulWeightTransposed(w).data.copyToFloatArray()

            assertContentEquals(viaPrimitive, viaTranspose, "$name: x.matmul(w.t()) must be x · Wᵀ, bit for bit")
        }
    }

    @Test
    fun `transposing twice gives the weight back`() {
        for ((name, geom, build) in encodings) {
            val (w, _) = weight(geom.first, geom.second, build)
            val back = w.t().t()
            assertEquals(w.shape, back.shape, "$name: shape after two transposes")
            assertSame(w.data, back.data, "$name: t().t() must be the original data, not a copy of it")
        }
    }

    @Test
    fun `the marker refuses to look like packed storage`() {
        for ((name, geom, build) in encodings) {
            val (w, _) = weight(geom.first, geom.second, build)
            val t = w.t()

            assertTrue(t.data is TransposedWeightTensorData<*, *>, "$name: transpose produced ${t.data::class.simpleName}")
            // The whole safety argument: a kernel that asks "are you packed bytes I can read?" must
            // hear no. The relabel this replaced said yes and handed over the wrong order (#973).
            assertFalse(
                t.data is PackedBlockStorage,
                "$name: the marker must not be PackedBlockStorage — that is how the old relabel fed kernels garbage",
            )
        }
    }

    @Test
    fun `reading the marker mirrors the weight with the indices swapped`() {
        val (w, inDim) = weight(32, 34) { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> }
        val t = w.t()
        assertEquals(Shape(inDim, outDim), t.shape)
        // Read through a star projection: the fixture is really `TensorData<FP32, Byte>` wearing a
        // `<FP32, Float>` cast (the repo's fixture idiom), and asking it for a Float is a real
        // ClassCastException on Kotlin/Wasm, which checks the cast the JVM erases. The element
        // values are quantized codes either way — that is the point of the assertion.
        val weightData: TensorData<*, *> = w.data
        val markerData: TensorData<*, *> = t.data
        for (o in 0 until outDim) {
            for (i in 0 until inDim step 37) {
                assertEquals(weightData[o, i], markerData[i, o], "element [$i, $o] of Wᵀ must be [$o, $i] of W")
            }
        }
    }

    @Test
    fun `repeated use of the same transposed weight stays correct`() {
        // The relayout behind the marker is cached per weight (#1096). Nothing may go stale.
        val (w, inDim) = weight(32, 34) { s, b -> Q8_0BlockTensorData(s, b) as TensorData<FP32, Float> }
        val x = activation(inDim)
        val first = x.matmul(w.t()).data.copyToFloatArray()
        repeat(5) {
            assertContentEquals(first, x.matmul(w.t()).data.copyToFloatArray(), "answer drifted on reuse")
        }
    }

    @Test
    fun `a dense weight is untouched by any of this`() {
        val inDim = 96
        val dense = ctx.fromFloatArray<FP32, Float>(
            Shape(outDim, inDim), FP32::class, FloatArray(outDim * inDim) { it * 0.01f },
        )
        val t = dense.t()
        assertEquals(Shape(inDim, outDim), t.shape)
        assertFalse(t.data is TransposedWeightTensorData<*, *>, "a dense transpose is a real transpose, not a marker")

        val x = activation(inDim)
        assertContentEquals(
            x.matmulWeightTransposed(dense).data.copyToFloatArray(),
            x.matmul(t).data.copyToFloatArray(),
            "dense: the two spellings already agreed and must keep agreeing",
        )
    }

    @Test
    fun `the answer matches dequantizing the weight and doing it the slow honest way`() {
        // Routing tests only prove the two spellings agree with each other. This one checks they
        // agree with the *matrix*: dequantize W, compute `sum_i x[i] * W[o, i]` in plain floats,
        // and compare. It is the shape of check that catches a wrong block order, which is the
        // failure mode this whole area exists because of (#968) — a relabel produces numbers that
        // look fine and are not these ones.
        for ((name, geom, build) in simpleEncodings) {
            val (w, inDim) = validWeight(name, geom.first, geom.second, build)
            val x = activation(inDim)

            val dequantized = (w.data as PackedBlockStorage).toFloatArray()   // [out, in], row-major
            val xs = x.data.copyToFloatArray()
            val expected = FloatArray(outDim) { o ->
                var acc = 0.0f
                for (i in 0 until inDim) acc += xs[i] * dequantized[o * inDim + i]
                acc
            }

            val actual = x.matmul(w.t()).data.copyToFloatArray()
            assertEquals(expected.size, actual.size, "$name: result length")
            for (o in expected.indices) {
                val tolerance = 1e-3f * maxOf(1.0f, abs(expected[o]))
                assertTrue(
                    abs(expected[o] - actual[o]) <= tolerance,
                    "$name: output[$o] was ${actual[o]}, the dequantized matrix says ${expected[o]}",
                )
            }

            // And the check has to be able to fail. Read the same blocks in the other order — which
            // is precisely the mistake a relabel makes — and require a different answer. Without
            // this the assertion above would pass vacuously on a fixture where the two orders
            // happen to coincide, which is how #968 survived its own tests.
            val blockElems = geom.first
            val misordered = FloatArray(dequantized.size)
            for (o in 0 until outDim) {
                for (b in 0 until blocksPerRow) {
                    val src = (b * outDim + o) * blockElems
                    val dst = (o * blocksPerRow + b) * blockElems
                    dequantized.copyInto(misordered, dst, src, src + blockElems)
                }
            }
            val wrong = FloatArray(outDim) { o ->
                var acc = 0.0f
                for (i in 0 until inDim) acc += xs[i] * misordered[o * inDim + i]
                acc
            }
            assertTrue(
                expected.indices.any { abs(expected[it] - wrong[it]) > 1e-3f * maxOf(1.0f, abs(expected[it])) },
                "$name: the two block orders give the same answer, so this fixture proves nothing",
            )
        }
    }
}
