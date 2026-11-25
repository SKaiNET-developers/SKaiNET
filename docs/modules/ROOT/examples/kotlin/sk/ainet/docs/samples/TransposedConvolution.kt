package sk.ainet.docs.samples

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * `convTranspose2d` on a 2x2 input. The regions are included into the operator doc partial
 * `partials/ops/tensorops/convtranspose2d.adoc`; `SamplesTest` runs them.
 */
object TransposedConvolution {

    private fun image(ctx: DirectCpuExecutionContext, shape: Shape, vararg values: Float): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, values)

    /** Stride 1: neighbouring kernel copies overlap and add up. */
    // tag::overlap[]
    fun overlap(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> {
        val x = image(ctx, Shape(1, 1, 2, 2), 1f, 2f, 3f, 4f)      // [N, C_in, H, W]
        val w = image(ctx, Shape(1, 1, 2, 2), 1f, 1f, 1f, 1f)      // [C_in, C_out / groups, kH, kW]

        // Each input value adds a copy of the kernel, scaled by that value, to the output.
        // (2 - 1) * 1 + (2 - 1) + 1 = 3 per axis:
        //   1  3  2
        //   4 10  6
        //   3  7  4
        return x.ops.convTranspose2d(x, w)
    }
    // end::overlap[]

    /** Stride 2 with a 2x2 kernel: the copies do not overlap, the image is upsampled by 2. */
    // tag::upsample[]
    fun upsample(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> {
        val x = image(ctx, Shape(1, 1, 2, 2), 1f, 2f, 3f, 4f)
        val w = image(ctx, Shape(1, 1, 2, 2), 1f, 1f, 1f, 1f)

        // (2 - 1) * 2 + (2 - 1) + 1 = 4 per axis:
        //   1 1 2 2
        //   1 1 2 2
        //   3 3 4 4
        //   3 3 4 4
        return x.ops.convTranspose2d(x, w, stride = 2 to 2)
    }
    // end::upsample[]
}
