package sk.ainet.lang.nn.optim

import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.Parameter
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.math.pow

/**
 * Adam optimizer (Adaptive Moment Estimation).
 *
 * Implements the Adam algorithm from "Adam: A Method for Stochastic Optimization"
 * (Kingma & Ba, 2014) with optional decoupled weight decay (AdamW).
 *
 * The update rule is:
 * ```
 * m_t = β1 * m_{t-1} + (1 - β1) * g_t
 * v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
 * m_hat = m_t / (1 - β1^t)
 * v_hat = v_t / (1 - β2^t)
 * θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + ε)
 * ```
 *
 * When [decoupledWeightDecay] is true (default), weight decay is applied directly
 * to the parameters (AdamW style) rather than added to the gradient (L2 regularization).
 *
 * @param lr Learning rate (default: 0.001)
 * @param beta1 Exponential decay rate for the first moment estimates (default: 0.9)
 * @param beta2 Exponential decay rate for the second moment estimates (default: 0.999)
 * @param epsilon Small constant for numerical stability (default: 1e-8)
 * @param weightDecay Weight decay coefficient (default: 0.0)
 * @param decoupledWeightDecay If true, uses AdamW-style decoupled weight decay (default: true)
 * @param amsgrad If true, uses the AMSGrad variant that maintains the maximum of all v_t (default: false)
 */
public class AdamOptimizer @kotlin.jvm.JvmOverloads constructor(
    /**
     * Learning rate. Mutable so learning-rate schedules (warmup, cosine
     * decay, …) can adjust it between steps without recreating the optimizer
     * and losing the moment estimates. See [LrSchedule].
     */
    public var lr: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-8,
    private val weightDecay: Double = 0.0,
    private val decoupledWeightDecay: Boolean = true,
    private val amsgrad: Boolean = false,
) : Optimizer {

    private data class Entry(
        val param: ModuleParameter<*, *>,
        val applyWeightDecay: Boolean,
        var m: Tensor<out DType, *>? = null,  // First moment estimate
        var v: Tensor<out DType, *>? = null,  // Second moment estimate
        var vMax: Tensor<out DType, *>? = null // Max of v for AMSGrad
    )

    private val params: MutableList<Entry> = mutableListOf()
    private var step: Int = 0

    override fun addParameter(param: Parameter, applyWeightDecay: Boolean) {
        addParameter(param.moduleParameter, applyWeightDecay)
    }

    override fun addParameter(param: ModuleParameter<*, *>, applyWeightDecay: Boolean) {
        params += Entry(param, applyWeightDecay)
    }

    override fun zeroGrad() {
        params.forEach { it.param.value.zeroGrad() }
    }

    @Suppress("UNCHECKED_CAST")
    override fun step() {
        step++

        // Bias correction terms
        val biasCorrection1 = 1.0 - beta1.pow(step)
        val biasCorrection2 = 1.0 - beta2.pow(step)

        for (e in params) {
            val p = e.param
            val tensor = p.value as Tensor<DType, Any?>
            val gradAny = tensor.grad as Tensor<DType, Any?>?
            if (!p.requiresGrad || gradAny == null) continue

            val ops = tensor.ops

            // Apply L2 regularization to gradient if not using decoupled weight decay
            val g = if (!decoupledWeightDecay && e.applyWeightDecay && weightDecay != 0.0) {
                val wdTerm = ops.mulScalar(tensor, weightDecay) as Tensor<DType, Any?>
                ops.add(gradAny, wdTerm) as Tensor<DType, Any?>
            } else gradAny

            // Update biased first moment estimate: m = β1 * m + (1 - β1) * g
            val mPrev = e.m as Tensor<DType, Any?>?
            val mNew = if (mPrev == null) {
                // Initialize m = (1 - β1) * g
                ops.mulScalar(g, 1.0 - beta1) as Tensor<DType, Any?>
            } else {
                // m = β1 * m_prev + (1 - β1) * g
                val scaledM = ops.mulScalar(mPrev, beta1) as Tensor<DType, Any?>
                val scaledG = ops.mulScalar(g, 1.0 - beta1) as Tensor<DType, Any?>
                ops.add(scaledM, scaledG) as Tensor<DType, Any?>
            }
            e.m = mNew

            // Update biased second moment estimate: v = β2 * v + (1 - β2) * g^2
            val vPrev = e.v as Tensor<DType, Any?>?
            val gSquared = ops.multiply(g, g) as Tensor<DType, Any?>
            val vNew = if (vPrev == null) {
                // Initialize v = (1 - β2) * g^2
                ops.mulScalar(gSquared, 1.0 - beta2) as Tensor<DType, Any?>
            } else {
                // v = β2 * v_prev + (1 - β2) * g^2
                val scaledV = ops.mulScalar(vPrev, beta2) as Tensor<DType, Any?>
                val scaledGSq = ops.mulScalar(gSquared, 1.0 - beta2) as Tensor<DType, Any?>
                ops.add(scaledV, scaledGSq) as Tensor<DType, Any?>
            }
            e.v = vNew

            // Compute bias-corrected estimates
            val mHat = ops.divScalar(mNew, biasCorrection1) as Tensor<DType, Any?>

            val vForDenom = if (amsgrad) {
                // AMSGrad: use max(v_t, v_{t-1}, ..., v_1)
                val vMaxPrev = e.vMax as Tensor<DType, Any?>?
                val vMaxNew = if (vMaxPrev == null) {
                    vNew
                } else {
                    // Element-wise max: we'll use a workaround since there's no max op
                    // max(a, b) = (a + b + |a - b|) / 2
                    // For simplicity, we'll just compare and use the larger value
                    // But since we don't have element-wise max, use a simpler approach:
                    // We can approximate by using the running max mechanism
                    elementwiseMax(ops, vMaxPrev, vNew)
                }
                e.vMax = vMaxNew
                ops.divScalar(vMaxNew, biasCorrection2) as Tensor<DType, Any?>
            } else {
                ops.divScalar(vNew, biasCorrection2) as Tensor<DType, Any?>
            }

            // Compute update: lr * m_hat / (sqrt(v_hat) + ε)
            val sqrtV = ops.sqrt(vForDenom) as Tensor<DType, Any?>
            val denom = ops.addScalar(sqrtV, epsilon) as Tensor<DType, Any?>
            val update = ops.divide(mHat, denom) as Tensor<DType, Any?>
            val scaledUpdate = ops.mulScalar(update, lr) as Tensor<DType, Any?>

            // Apply decoupled weight decay if enabled: θ = θ - lr * wd * θ
            val afterWeightDecay = if (decoupledWeightDecay && e.applyWeightDecay && weightDecay != 0.0) {
                val wdUpdate = ops.mulScalar(tensor, lr * weightDecay) as Tensor<DType, Any?>
                ops.subtract(tensor, wdUpdate) as Tensor<DType, Any?>
            } else tensor

            // Final update: θ = θ - scaled_update
            val newP = ops.subtract(afterWeightDecay, scaledUpdate) as Tensor<out DType, Any?>

            // Reassign parameter value
            @Suppress("UNCHECKED_CAST")
            (p as ModuleParameter<DType, Any?>).value = newP as Tensor<DType, Any?>
        }
    }

    /**
     * Element-wise maximum of two tensors.
     * Since TensorOps doesn't have a max operation, we compute:
     * max(a, b) = (a + b + abs(a - b)) / 2
     * And abs(x) = sqrt(x^2) for a differentiable approximation
     */
    @Suppress("UNCHECKED_CAST")
    private fun elementwiseMax(ops: sk.ainet.lang.tensor.ops.TensorOps, a: Tensor<DType, Any?>, b: Tensor<DType, Any?>): Tensor<DType, Any?> {
        val sum = ops.add(a, b) as Tensor<DType, Any?>
        val diff = ops.subtract(a, b) as Tensor<DType, Any?>
        val diffSquared = ops.multiply(diff, diff) as Tensor<DType, Any?>
        val absDiff = ops.sqrt(diffSquared) as Tensor<DType, Any?>
        val sumWithAbs = ops.add(sum, absDiff) as Tensor<DType, Any?>
        return ops.divScalar(sumWithAbs, 2.0) as Tensor<DType, Any?>
    }

    /**
     * Resets the optimizer state (moment estimates and step counter).
     * Useful when starting training from scratch with the same optimizer instance.
     */
    public fun reset() {
        step = 0
        params.forEach { e ->
            e.m = null
            e.v = null
            e.vMax = null
        }
    }
}

/**
 * Factory function for Adam optimizer.
 *
 * @param lr Learning rate (default: 0.001)
 * @param beta1 Exponential decay rate for first moment (default: 0.9)
 * @param beta2 Exponential decay rate for second moment (default: 0.999)
 * @param epsilon Numerical stability constant (default: 1e-8)
 * @param weightDecay Weight decay coefficient (default: 0.0)
 * @param decoupledWeightDecay Use AdamW-style decoupled weight decay (default: true)
 * @param amsgrad Use AMSGrad variant (default: false)
 */
public fun adam(
    lr: Double = 0.001,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8,
    weightDecay: Double = 0.0,
    decoupledWeightDecay: Boolean = true,
    amsgrad: Boolean = false,
): Optimizer = AdamOptimizer(lr, beta1, beta2, epsilon, weightDecay, decoupledWeightDecay, amsgrad)

/**
 * Factory function for AdamW optimizer (Adam with decoupled weight decay).
 * This is equivalent to adam() with decoupledWeightDecay=true.
 *
 * @param lr Learning rate (default: 0.001)
 * @param beta1 Exponential decay rate for first moment (default: 0.9)
 * @param beta2 Exponential decay rate for second moment (default: 0.999)
 * @param epsilon Numerical stability constant (default: 1e-8)
 * @param weightDecay Weight decay coefficient (default: 0.01)
 */
public fun adamw(
    lr: Double = 0.001,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8,
    weightDecay: Double = 0.01,
): Optimizer = AdamOptimizer(lr, beta1, beta2, epsilon, weightDecay, decoupledWeightDecay = true, amsgrad = false)
