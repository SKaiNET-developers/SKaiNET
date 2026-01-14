package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.*
import sk.ainet.lang.types.DType


/**
 * A simple layer that flattens an input tensor into a 1D tensor.
 * This layer has no parameters and simply reshapes the input.
 */
public class Flatten<T : DType, V>(
    private val startDim: Int = 1,
    private val endDim: Int = -1,
    override val name: String = "Flatten"
) : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        input.flatten(startDim, endDim)
}
