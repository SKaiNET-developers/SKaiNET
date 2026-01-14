package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.topology.Parameter
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType

public abstract class Module<T : DType, V> : ModuleNode {

    public abstract override val name: String

    public abstract val modules: List<Module<T, V>>

    public open fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val boundInput = input.bind(ctx)
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, boundInput) {
            onForward(boundInput, ctx)
        }
    }

    protected open fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        return input
    }

    // ModuleNode implementation
    override val id: String get() = name
    override var path: String? = null
    override val children: List<ModuleNode>
        get() = modules.map { it as ModuleNode }
    @Suppress("UNCHECKED_CAST")
    override val params: List<ModuleParameter<*, *>>
        get() = when (this) {
            is ModuleParameters<*, *> -> (this as ModuleParameters<Any?, Any?>).params as List<ModuleParameter<*, *>>
            else -> emptyList()
        }

    /**
     * Collect all trainable parameters for this module and its subtree.
     * Trainable is defined by ModuleParameter.requiresGrad flag.
     */
    public open fun trainableParameters(): List<Parameter> {
        val own = params.filter { it.requiresGrad }.map { Parameter(it.name, it) }
        val childParams = modules.flatMap { it.trainableParameters() }
        return own + childParams
    }

    /** Zero-out accumulated gradients for all trainable parameters in this module subtree. */
    public fun zeroGrad() {
        trainableParameters().forEach { p -> p.value.zeroGrad() }
    }
}

