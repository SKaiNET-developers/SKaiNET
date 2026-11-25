package sk.ainet.docs.samples

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32

/**
 * Normalization layers in the `sequential { }` DSL. The regions are included into
 * `how-to/normalization-layers.adoc`; `SamplesTest` runs them.
 */
object NormalizationLayers {

    /** A small convolutional block normalized with GroupNorm: 8 channels in 4 groups of 2. */
    // tag::groupnorm[]
    fun convBlock(ctx: DirectCpuExecutionContext) =
        sequential<FP32, Float>(ctx) {
            input(intArrayOf(3, 16, 16))                      // one RGB image, 16x16
            conv2d(outChannels = 8, kernelSize = 3 to 3, padding = 1 to 1)
            groupNorm(numGroups = 4, numChannels = 8)         // statistics per sample and group
            activation { it.relu() }
        }
    // end::groupnorm[]

    /** Runs the block on a batch of one: batch size does not matter to GroupNorm. */
    // tag::forward[]
    fun forwardOne(pixels: FloatArray): Tensor<FP32, Float> {
        val ctx = DirectCpuExecutionContext.create()
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 3, 16, 16), FP32::class, pixels)
        return convBlock(ctx).forward(input, ctx)             // [1, 8, 16, 16]
    }
    // end::forward[]
}
