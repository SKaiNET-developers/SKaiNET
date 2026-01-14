package sk.ainet.lang.tensor.operators

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Lightweight wrapper that binds a Tensor to a specific TensorOps implementation.
 * Useful to evaluate operator overloads with a desired backend (CPU/GPU/etc.).
 */
public class OpsBoundTensor<T : DType, V>(
    override val data: TensorData<T, V>,
    override val dtype: KClass<T>,
    private val opsRef: TensorOps,
    override val gradState: GradState<T, V> = GradState()
) : Tensor<T, V> {
    override val ops: TensorOps
        get() = opsRef
}

/**
 * Returns a Tensor that uses the provided ops for subsequent operations.
 */
public fun <T : DType, V> Tensor<T, V>.withOps(ops: TensorOps): Tensor<T, V> =
    OpsBoundTensor(this.data, this.dtype, ops, this.gradState)

/**
 * Binds this tensor to the operations of the given execution context.
 */
public fun <T : DType, V> Tensor<T, V>.bind(ctx: sk.ainet.context.ExecutionContext): Tensor<T, V> =
    this.withOps(ctx.ops)
