package sk.ainet.app.notebook.tools

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Nearest-neighbor upscaling utilities.
 *
 * Notes:
 * - Only integer scale factors are supported (scale >= 1)
 * - Implemented for Float FP32 tensors which is a common case for MNIST-like data
 * - Supported layouts: HW (grayscale), CHW (C,H,W)
 */
public object UpsampleNearestNeighbor {

    /**
     * Generic nearest-neighbor upscale supporting common image tensor layouts.
     * Supported layouts: HW, CHW, HWC, NHWC, NCHW.
     */
    public fun <T : DType, V> upscaleNearest(
        input: Tensor<T, V>,
        scale: Int,
        layout: Layout,
        ctx: ExecutionContext
    ): Tensor<T, V> {
        require(scale >= 1) { "Scale must be an integer >= 1, got $scale" }

        return when (layout) {
            Layout.HW -> {
                require(input.shape.rank == 2) { "HW layout expects rank-2 tensor (H,W), got rank ${input.shape.rank} with shape ${input.shape}" }
                val h = input.shape[0]
                val w = input.shape[1]
                val oh = h * scale
                val ow = w * scale
                val out = ctx.zeros<T, V>(Shape(oh, ow), input.dtype)
                for (yOut in 0 until oh) {
                    val yIn = yOut / scale
                    for (xOut in 0 until ow) {
                        val xIn = xOut / scale
                        out.data[yOut, xOut] = input.data[yIn, xIn]
                    }
                }
                out
            }

            Layout.CHW -> {
                require(input.shape.rank == 3) { "CHW layout expects rank-3 tensor (C,H,W), got rank ${input.shape.rank} with shape ${input.shape}" }
                val c = input.shape[0]
                val h = input.shape[1]
                val w = input.shape[2]
                val oh = h * scale
                val ow = w * scale
                val out = ctx.zeros<T, V>(Shape(c, oh, ow), input.dtype)
                for (ch in 0 until c) {
                    for (yOut in 0 until oh) {
                        val yIn = yOut / scale
                        for (xOut in 0 until ow) {
                            val xIn = xOut / scale
                            out.data[ch, yOut, xOut] = input.data[ch, yIn, xIn]
                        }
                    }
                }
                out
            }

            Layout.HWC -> {
                require(input.shape.rank == 3) { "HWC layout expects rank-3 tensor (H,W,C), got rank ${input.shape.rank} with shape ${input.shape}" }
                val h = input.shape[0]
                val w = input.shape[1]
                val c = input.shape[2]
                val oh = h * scale
                val ow = w * scale
                val out = ctx.zeros<T, V>(Shape(oh, ow, c), input.dtype)
                for (yOut in 0 until oh) {
                    val yIn = yOut / scale
                    for (xOut in 0 until ow) {
                        val xIn = xOut / scale
                        for (ch in 0 until c) {
                            out.data[yOut, xOut, ch] = input.data[yIn, xIn, ch]
                        }
                    }
                }
                out
            }

            Layout.NHWC -> {
                require(input.shape.rank == 4) { "NHWC layout expects rank-4 tensor (N,H,W,C), got rank ${input.shape.rank} with shape ${input.shape}" }
                val n = input.shape[0]
                val h = input.shape[1]
                val w = input.shape[2]
                val c = input.shape[3]
                val oh = h * scale
                val ow = w * scale
                val out = ctx.zeros<T, V>(Shape(n, oh, ow, c), input.dtype)
                for (b in 0 until n) {
                    for (yOut in 0 until oh) {
                        val yIn = yOut / scale
                        for (xOut in 0 until ow) {
                            val xIn = xOut / scale
                            for (ch in 0 until c) {
                                out.data[b, yOut, xOut, ch] = input.data[b, yIn, xIn, ch]
                            }
                        }
                    }
                }
                out
            }

            Layout.NCHW -> {
                require(input.shape.rank == 4) { "NCHW layout expects rank-4 tensor (N,C,H,W), got rank ${input.shape.rank} with shape ${input.shape}" }
                val n = input.shape[0]
                val c = input.shape[1]
                val h = input.shape[2]
                val w = input.shape[3]
                val oh = h * scale
                val ow = w * scale
                val out = ctx.zeros<T, V>(Shape(n, c, oh, ow), input.dtype)
                for (b in 0 until n) {
                    for (ch in 0 until c) {
                        for (yOut in 0 until oh) {
                            val yIn = yOut / scale
                            for (xOut in 0 until ow) {
                                val xIn = xOut / scale
                                out.data[b, ch, yOut, xOut] = input.data[b, ch, yIn, xIn]
                            }
                        }
                    }
                }
                out
            }
        }
    }
}