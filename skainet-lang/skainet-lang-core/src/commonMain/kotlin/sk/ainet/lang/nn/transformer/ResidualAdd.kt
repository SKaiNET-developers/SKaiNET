package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Residual (skip) connection module.
 *
 * Saves the input before the preceding sublayer and adds it to the sublayer's output.
 * Used as: norm → sublayer → residual() which computes output = sublayerOutput + savedInput
 *
 * In the network DSL, `residual()` marks the end of a residual block. The input to the
 * block is captured when the residual module is first encountered, and added to the
 * block's final output.
 *
 * @param name module name
 */
public class ResidualAdd<T : DType, V>(
    override val name: String = "ResidualAdd"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = emptyList()

    // The saved "skip" tensor — set by the Sequential/MLP container
    // before executing the residual sublayer.
    public var savedInput: Tensor<T, V>? = null

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val skip = savedInput ?: return input
        savedInput = null // consume
        return ctx.ops.add(input, skip)
    }
}
