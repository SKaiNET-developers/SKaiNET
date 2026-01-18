package sk.ainet.app.notebook.tools

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UpsampleNearestNeighborTest {

    private val ctx = DefaultDataExecutionContext()

    private fun chessboardHW(h: Int, w: Int, v0: Float = 0f, v1: Float = 1f): Tensor<FP32, Float> {
        val data = ctx.zeros<FP32, Float>(Shape(h, w), FP32::class)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val v = if ((x + y) % 2 == 0) v0 else v1
                data.data[y, x] = v
            }
        }
        return data
    }

    private fun chessboardCHW(h: Int, w: Int, v0: Float = 0f, v1: Float = 1f): Tensor<FP32, Float> {
        val data = ctx.zeros<FP32, Float>(Shape(1, h, w), FP32::class)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val v = if ((x + y) % 2 == 0) v0 else v1
                data.data[0, y, x] = v
            }
        }
        return data
    }

    private fun chessboardHWInt(h: Int, w: Int, v0: Int = 0, v1: Int = 1): Tensor<Int32, Int> {
        val data = ctx.zeros<Int32, Int>(Shape(h, w), Int32::class)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val v = if ((x + y) % 2 == 0) v0 else v1
                data.data[y, x] = v
            }
        }
        return data
    }

    @Test
    fun testNearestUpscaleHW_2x() {
        val input = chessboardHW(4, 4)
        val scale = 2
        val out = UpsampleNearestNeighbor.upscaleNearest(input, scale, Layout.HW, ctx)

        assertEquals(Shape(8, 8), out.shape)

        // Verify that each input pixel becomes a 2x2 block with same value
        for (y in 0 until 4) {
            for (x in 0 until 4) {
                val v = input.data[y, x]
                val y0 = y * scale
                val x0 = x * scale
                val block = listOf(
                    out.data[y0, x0], out.data[y0, x0 + 1],
                    out.data[y0 + 1, x0], out.data[y0 + 1, x0 + 1]
                )
                assertTrue(block.all { it == v }, "Block at ($y,$x) not replicated correctly: $block vs $v")
            }
        }
    }

    @Test
    fun testNearestUpscaleCHW_3x_mnistLike() {
        // MNIST-like: C=1, HxW
        val input = chessboardCHW(6, 6)
        val scale = 3
        val out = UpsampleNearestNeighbor.upscaleNearest(input, scale, Layout.CHW, ctx)

        assertEquals(Shape(1, 18, 18), out.shape)

        for (y in 0 until 6) {
            for (x in 0 until 6) {
                val v = input.data[0, y, x]
                val y0 = y * scale
                val x0 = x * scale
                for (dy in 0 until scale) {
                    for (dx in 0 until scale) {
                        assertEquals(v, out.data[0, y0 + dy, x0 + dx], "Mismatch at (${y0 + dy}, ${x0 + dx})")
                    }
                }
            }
        }
    }

    @Test
    fun testNearestUpscaleHW_Int32_2x() {
        val input = chessboardHWInt(3, 5)
        val scale = 2
        val out = UpsampleNearestNeighbor.upscaleNearest(input, scale, Layout.HW, ctx)

        assertEquals(Shape(6, 10), out.shape)

        for (y in 0 until 3) {
            for (x in 0 until 5) {
                val v = input.data[y, x]
                val y0 = y * scale
                val x0 = x * scale
                val block = listOf(
                    out.data[y0, x0], out.data[y0, x0 + 1],
                    out.data[y0 + 1, x0], out.data[y0 + 1, x0 + 1]
                )
                assertTrue(block.all { it == v }, "Int32 block at ($y,$x) not replicated correctly: $block vs $v")
            }
        }
    }
}
