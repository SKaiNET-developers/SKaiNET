package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Runs [inner], a module built in another dtype, inside a network of dtype [TOuter]:
 * the input is cast to [innerType], [inner] runs, and the result is cast back to
 * [outerType] (SKEEP-008, #1433).
 *
 * The casts use [sk.ainet.lang.tensor.ops.TensorOps.convert], which returns its input
 * unchanged when the dtypes already match, so a boundary between equal dtypes costs
 * nothing. `convert` is recorded by the tracing wrappers and lowers to
 * `stablehlo.convert`, so the boundary stays visible in traces and exports.
 *
 * `convert` is not differentiable (SKEEP-008 phase A3 lifts this), so gradients stop at
 * the boundary: the wrapped layers do not train.
 */
public class PrecisionBoundary<TOuter : DType, TInner : DType, V>(
    override val name: String,
    private val inner: Module<TInner, V>,
    private val innerType: TInner,
    private val outerType: TOuter,
) : Module<TOuter, V>() {

    @Suppress("UNCHECKED_CAST")
    override val modules: List<Module<TOuter, V>>
        // The dtype parameter is erased at runtime; the inner module is listed so that
        // reflection, parameter collection and checkpointing still see its subtree.
        get() = listOf(inner as Module<TOuter, V>)

    override fun onForward(input: Tensor<TOuter, V>, ctx: ExecutionContext): Tensor<TOuter, V> {
        val x = ctx.ops.convert(input, innerType)
        return ctx.ops.convert(inner.forward(x, ctx), outerType)
    }
}
