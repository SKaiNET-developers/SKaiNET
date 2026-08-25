package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TransposedWeightTensorData
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * #1096 (#973): `x · Wᵀ` as a primitive, and `transpose` on a packed weight as an error.
 *
 * The old shape — `matmul(x, transpose(w))` — copied the whole weight on every call, and
 * `transpose(transpose(w))` was not `w`. Both are gone: the product is asked for directly, the
 * relayout happens once per weight, and `transpose` refuses rather than lying.
 */
class MatmulWeightTransposedTest {

    private val ctx = DirectCpuExecutionContext()
    private val outDim = 4
    private val inDim = 96          // three Q8_0 blocks per row: the case where block order matters

    @Suppress("UNCHECKED_CAST")
    private fun weight(): Tensor<FP32, Float> {
        val blocks = outDim * (inDim / 32)
        val bytes = ByteArray(blocks * 34)
        var seed = 11
        for (b in 0 until blocks) {
            val base = b * 34
            bytes[base] = 0x00; bytes[base + 1] = 0x3C      // fp16 scale 1.0
            for (i in 0 until 32) {
                seed = seed * 1103515245 + 12345
                bytes[base + 2 + i] = ((seed ushr 16) % 17 - 8).toByte()
            }
        }
        val data = Q8_0BlockTensorData(Shape(outDim, inDim), bytes)
        return ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
    }

    private fun activation(): Tensor<FP32, Float> =
        ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, FloatArray(inDim) { (it % 7) * 0.125f })

    @Test
    fun `the primitive agrees with the relayout then matmul it replaces`() {
        val w = weight()
        val x = activation()
        val viaPrimitive = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
        val viaRelayout = ctx.ops.matmul(x, ctx.ops.relayoutPackedWeightForKernels(w)).data.copyToFloatArray()
        assertContentEquals(viaRelayout, viaPrimitive, "the primitive must compute exactly what the old path did")
    }

    @Test
    fun `a weight is relayouted once however many times it is used`() {
        val w = weight()
        val x = activation()
        val first = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
        repeat(5) {
            assertContentEquals(first, ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray())
        }
        // and the answer keeps matching the explicit relayout, so the cache is not stale
        assertContentEquals(
            ctx.ops.matmul(x, ctx.ops.relayoutPackedWeightForKernels(w)).data.copyToFloatArray(),
            first,
        )
    }

    @Test
    fun `transpose of a packed weight is the transposed-weight marker`() {
        // #1108 replaced #1096's refusal: `t()` no longer throws, it returns `Wᵀ` unmaterialized.
        val t = ctx.ops.transpose(weight())
        assertTrue(t.shape == Shape(inDim, outDim), "the shape is the transpose: ${t.shape}")
        assertTrue(t.data is TransposedWeightTensorData<*, *>, "and the data says so: ${t.data::class.simpleName}")
    }

    @Test
    fun `a dense weight transposes as it always did`() {
        val dense: Tensor<FP32, Float> =
            ctx.fromFloatArray<FP32, Float>(Shape(outDim, inDim), FP32::class, FloatArray(outDim * inDim) { it * 0.01f })
        val t = ctx.ops.transpose(dense)
        assertTrue(t.shape == Shape(inDim, outDim))
        val x = activation()
        val viaPrimitive = ctx.ops.matmulWeightTransposed(x, dense).data.copyToFloatArray()
        val viaTranspose = ctx.ops.matmul(x, t).data.copyToFloatArray()
        for (i in viaPrimitive.indices) {
            assertTrue(abs(viaPrimitive[i] - viaTranspose[i]) < 1e-4f, "[$i]: ${viaPrimitive[i]} vs ${viaTranspose[i]}")
        }
    }

    @Test
    fun `the extension is the ops call for packed and dense alike`() {
        val x = activation()

        val packed = weight()
        assertContentEquals(
            ctx.ops.matmulWeightTransposed(x, packed).data.copyToFloatArray(),
            x.matmulWeightTransposed(packed).data.copyToFloatArray(),
            "x.matmulWeightTransposed(w) must be ops.matmulWeightTransposed(x, w) for a packed weight",
        )

        val dense: Tensor<FP32, Float> =
            ctx.fromFloatArray<FP32, Float>(Shape(outDim, inDim), FP32::class, FloatArray(outDim * inDim) { it * 0.01f })
        assertContentEquals(
            ctx.ops.matmulWeightTransposed(x, dense).data.copyToFloatArray(),
            x.matmulWeightTransposed(dense).data.copyToFloatArray(),
            "and for a dense one",
        )
    }
}
