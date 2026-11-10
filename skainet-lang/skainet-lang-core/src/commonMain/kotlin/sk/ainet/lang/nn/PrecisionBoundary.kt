package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Runs [inner], a module built in another dtype, inside a network of dtype [TOuter]:
 * the input is cast to [innerType], [inner] runs, and the result is cast back to
 * [outerType] (SKEEP-008, #1433).
 *
 * The casts use [sk.ainet.lang.tensor.ops.TensorOps.convert]. A tensor that already has
 * the target dtype is passed through without calling it, so a boundary between equal
 * dtypes costs nothing and leaves no cast in a trace. Otherwise `convert` is recorded by
 * the tracing wrappers and lowers to `stablehlo.convert`, so the boundary stays visible
 * in traces and exports.
 *
 * A tracing context marks every op recorded inside the wrapped module with
 * `DTypePolicy.Prefer(innerType)`, so `DTypeConstraintResolutionPass` reports a boundary
 * whose ops a later pass retyped.
 *
 * `convert` is differentiable between floating-point dtypes, so gradients cross the
 * boundary and the wrapped layers train. The gradient is cast like the values: it is
 * rounded to [innerType] where it enters the wrapped module and widened back to
 * [outerType] where it leaves. There is no FP32 master copy of the weights and no loss
 * scaling; that is automatic mixed precision, which this module does not do.
 */
public class PrecisionBoundary<TOuter : DType, TInner : DType, V>(
    override val name: String,
    private val inner: Module<TInner, V>,
    /** The dtype the wrapped module's weights and computation use. */
    public val innerType: TInner,
    /** The dtype of the surrounding network, which the boundary's input and output have. */
    public val outerType: TOuter,
) : Module<TOuter, V>() {

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<TOuter, V>>
        // The dtype parameter is erased at runtime; the inner module is listed so that
        // reflection, parameter collection and checkpointing still see its subtree.
        get() = listOf(inner as Module<TOuter, V>)

    override fun onForward(input: Tensor<TOuter, V>, ctx: ExecutionContext): Tensor<TOuter, V> {
        val x = cast(input, innerType, ctx)
        return cast(inner.forward(x, ctx), outerType, ctx)
    }

    // A tracing wrapper records every `convert` call, also the one that returns its input,
    // so the same-dtype case is skipped here rather than left to the backend.
    @Suppress("UNCHECKED_CAST")
    private fun <TFrom : DType, TTo : DType> cast(
        tensor: Tensor<TFrom, V>,
        target: TTo,
        ctx: ExecutionContext,
    ): Tensor<TTo, V> =
        if (tensor.dtype == target::class) tensor as Tensor<TTo, V> else ctx.ops.convert(tensor, target)
}
