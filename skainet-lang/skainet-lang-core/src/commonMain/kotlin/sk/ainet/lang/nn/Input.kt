package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public class Input<T : DType, V>(
    override val name: String = "Input",
    public val requiresGrad: Boolean = false
) : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = emptyList()


    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        input.gradState.requiresGrad = requiresGrad
        return input
    }
}