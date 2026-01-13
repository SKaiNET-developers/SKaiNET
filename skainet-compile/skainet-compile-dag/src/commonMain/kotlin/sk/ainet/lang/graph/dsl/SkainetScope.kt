package sk.ainet.lang.graph.dsl

import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.DataContextDsl
import sk.ainet.lang.tensor.dsl.DataDefinitionContextDslImpl
import sk.ainet.tape.ExecutionTape
import kotlin.reflect.KClass

/**
 * DSL Scope for Skainet that wraps a GraphExecutionContext and provides
 * a more idiomatic way to interact with tensors and training loops.
 */
public class SkainetScope(
    public val ctx: DefaultGraphExecutionContext
) : DataContextDsl by DataDefinitionContextDslImpl(ctx) {

    /**
     * Mark a tensor as requiring gradients.
     */
    public fun <T : sk.ainet.lang.types.DType, V> Tensor<T, V>.withRequiresGrad(flag: Boolean = true): Tensor<T, V> {
        this.gradState.requiresGrad = flag
        return this
    }

    /**
     * Create a tensor with specific shape and values.
     */
    @Deprecated("Use tensor { shape(...) { from(...) } } or similar DSL", ReplaceWith("tensor(dtype) { shape(shape) { from(*values) } }"))
    public fun <T : sk.ainet.lang.types.DType, V> tensorOf(
        shape: Shape,
        dtype: KClass<T>,
        vararg values: Float
    ): Tensor<T, V> {
        return ctx.fromFloatArray(shape, dtype, values)
    }

    /**
     * High-level recording block.
     */
    public fun <R> record(block: SkainetScope.() -> R): Pair<ExecutionTape?, R> {
        return ctx.record {
            this@SkainetScope.block()
        }
    }

    /**
     * Encapsulated training step.
     */
    public fun trainStep(
        optimizer: Optimizer,
        vararg params: ModuleParameter<*, *>,
        block: SkainetScope.() -> Tensor<*, *>
    ): Tensor<*, *> {
        val (tape, loss) = record(block)
        
        ctx.backward(targets = listOf(loss), sources = params.map { it.value })
        
        params.forEach { optimizer.addParameter(it) }
        optimizer.step()
        // Grad is zeroed by the optimizer internally if implemented that way,
        // but let's keep it here to be explicit as per previous requirement.
        optimizer.zeroGrad()
        
        return loss
    }
}

/**
 * Entry point for the Skainet DSL.
 */
public fun skainet(ctx: DefaultGraphExecutionContext, block: SkainetScope.() -> Unit) {
    SkainetScope(ctx).block()
}
