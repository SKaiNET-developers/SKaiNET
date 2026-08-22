package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Simple tensor implementation with void Ops as default for the DSL
 */
public class VoidOpsTensor<T : DType, V>(
    override val data: TensorData<T, V>,
    override val dtype: KClass<T>,
    override val gradState: GradState<T, V> = GradState()
) : Tensor<T, V>, TensorIdBearer {
    override val ops: TensorOps
        get() = VoidTensorOps()

    override var id: TensorId? = null
}
