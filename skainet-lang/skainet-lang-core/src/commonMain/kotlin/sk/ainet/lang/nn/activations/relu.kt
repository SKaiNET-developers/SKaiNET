package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

import sk.ainet.lang.tensor.relu

public class ReLU<T : DType, V>(override val name: String = "ReLU") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> = input.relu()
}

