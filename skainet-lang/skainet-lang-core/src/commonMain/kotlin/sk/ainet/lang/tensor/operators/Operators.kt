package sk.ainet.lang.tensor.operators

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.TensorIdBearer
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Lightweight wrapper that binds a Tensor to a specific TensorOps implementation.
 * Useful to evaluate operator overloads with a desired backend (CPU/GPU/etc.).
 */
public class OpsBoundTensor<T : DType, V>(
    public val origin: Tensor<T, V>,
    private val opsRef: TensorOps,
) : Tensor<T, V>, TensorIdBearer {
    override val data: TensorData<T, V> get() = origin.data
    override val dtype: KClass<T> get() = origin.dtype
    override val gradState: GradState<T, V> get() = origin.gradState
    override val ops: TensorOps get() = opsRef

    private var localId: TensorId? = null

    /** The origin's id when it carries one (ids survive re-binding to another context), else a local one. */
    override var id: TensorId?
        get() = origin.id ?: localId
        set(value) {
            val o = origin
            if (o is TensorIdBearer) o.id = value else localId = value
        }

    override fun accumulateGrad(g: Tensor<T, V>) {
        origin.accumulateGrad(g)
    }

    override fun zeroGrad() {
        origin.zeroGrad()
    }

    public companion object {
        public fun <T : DType, V> fromData(data: TensorData<T, V>, dtype: KClass<T>, ops: TensorOps): OpsBoundTensor<T, V> {
            val origin = object : Tensor<T, V>, TensorIdBearer {
                override val data: TensorData<T, V> = data
                override val dtype: KClass<T> = dtype
                override val ops: TensorOps = ops
                override val gradState: GradState<T, V> = GradState()
                override var id: TensorId? = null
            }
            return OpsBoundTensor(origin, ops)
        }
    }
}

/**
 * Returns a Tensor that uses the provided ops for subsequent operations.
 */
public fun <T : DType, V> Tensor<T, V>.withOps(ops: TensorOps): Tensor<T, V> =
    OpsBoundTensor(this, ops)

/**
 * Binds this tensor to the operations of the given execution context.
 */
public fun <T : DType, V> Tensor<T, V>.bind(ctx: sk.ainet.context.ExecutionContext): Tensor<T, V> =
    this.withOps(ctx.ops)
