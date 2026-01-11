package sk.ainet.lang.nn.optim

import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Stochastic Gradient Descent optimizer with optional momentum and weight decay.
 */
public class SgdOptimizer(
    private val lr: Double,
    private val momentum: Double = 0.0,
    private val weightDecay: Double = 0.0,
) : Optimizer {

    private data class Entry(
        val param: ModuleParameter<*, *>,
        val applyWeightDecay: Boolean,
        var momentumBuf: Tensor<out DType, *>? = null
    )

    private val params: MutableList<Entry> = mutableListOf()

    override fun addParameter(param: ModuleParameter<*, *>, applyWeightDecay: Boolean) {
        params += Entry(param, applyWeightDecay, null)
    }

    override fun zeroGrad() {
        params.forEach { it.param.value.zeroGrad() }
    }

    @Suppress("UNCHECKED_CAST")
    override fun step() {
        for (e in params) {
            val p = e.param
            val tensor = p.value as Tensor<DType, Any?>
            val gradAny = tensor.grad as Tensor<DType, Any?>?
            if (!p.requiresGrad || gradAny == null) continue

            val ops = tensor.ops

            // Optionally add weight decay to gradient: grad += wd * p
            val g = if (e.applyWeightDecay && weightDecay != 0.0) {
                val wdTerm = ops.mulScalar(tensor, weightDecay) as Tensor<DType, Any?>
                ops.add(gradAny, wdTerm) as Tensor<DType, Any?>
            } else gradAny

            val update = if (momentum != 0.0) {
                val prev = e.momentumBuf
                val v = if (prev == null) {
                    // v = g
                    g
                } else {
                    // v = momentum * v + g
                    val mv = ops.mulScalar(prev as Tensor<DType, Any?>, momentum)
                    ops.add(mv as Tensor<DType, Any?>, g) as Tensor<DType, Any?>
                }
                e.momentumBuf = v
                v
            } else g

            // p = p - lr * update
            val scaled = ops.mulScalar(update, lr)
            val newP = ops.subtract(tensor, scaled as Tensor<DType, Any?>) as Tensor<out DType, Any?>

            // Reassign parameter value; keep requiresGrad flag via ModuleParameter.sync
            @Suppress("UNCHECKED_CAST")
            (p as ModuleParameter<DType, Any?>).value = newP as Tensor<DType, Any?>
        }
    }
}

/**
 * Small factory for SGD optimizer.
 */
public fun sgd(lr: Double, momentum: Double = 0.0, weightDecay: Double = 0.0): Optimizer =
    SgdOptimizer(lr, momentum, weightDecay)
