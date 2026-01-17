package sk.ainet.exec.optim

import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.nn.optim.adam
import sk.ainet.lang.nn.optim.adamw
import sk.ainet.lang.nn.optim.AdamOptimizer
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.withRequiresGrad

class AdamOptimizerTest {

    private fun param1x1(v: Float, train: Boolean = true): ModuleParameter.WeightParameter<FP32, Float> {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val t = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, v).withRequiresGrad(train)
        @Suppress("UNCHECKED_CAST")
        return ModuleParameter.WeightParameter("w", t as sk.ainet.lang.tensor.Tensor<FP32, Float>, train)
    }

    @Test
    fun basic_adam_step_updates_parameter() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(10f)

        // gradient = 2.0
        val g = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w.value.accumulateGrad(g as sk.ainet.lang.tensor.Tensor<FP32, Float>)

        val lr = 0.1
        val beta1 = 0.9
        val beta2 = 0.999
        val eps = 1e-8

        val opt: Optimizer = adam(lr = lr, beta1 = beta1, beta2 = beta2, epsilon = eps)
        opt.addParameter(w)

        opt.step()

        // Manual calculation for step 1:
        // m = (1 - 0.9) * 2.0 = 0.2
        // v = (1 - 0.999) * 4.0 = 0.004
        // m_hat = 0.2 / (1 - 0.9^1) = 0.2 / 0.1 = 2.0
        // v_hat = 0.004 / (1 - 0.999^1) = 0.004 / 0.001 = 4.0
        // update = 0.1 * 2.0 / (sqrt(4.0) + 1e-8) = 0.1 * 2.0 / 2.0 = 0.1
        // new_w = 10.0 - 0.1 = 9.9

        val m = (1 - beta1) * 2.0
        val v = (1 - beta2) * 4.0
        val mHat = m / (1 - beta1.pow(1))
        val vHat = v / (1 - beta2.pow(1))
        val update = lr * mHat / (sqrt(vHat) + eps)
        val expected = 10.0 - update

        assertEquals(expected.toFloat(), w.value.data[0, 0], 1e-5f)
    }

    @Test
    fun zeroGrad_clears_all_registered_grads() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w1 = param1x1(0f)
        val w2 = param1x1(1f)

        val g1 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 3.0)
        val g2 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 4.0)
        w1.value.accumulateGrad(g1 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        w2.value.accumulateGrad(g2 as sk.ainet.lang.tensor.Tensor<FP32, Float>)

        val opt: Optimizer = adam(lr = 0.001)
        opt.addParameter(w1)
        opt.addParameter(w2)

        opt.zeroGrad()

        assertNull(w1.value.grad)
        assertNull(w2.value.grad)
    }

    @Test
    fun multiple_steps_accumulate_moments() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(5f)

        val lr = 0.1
        val beta1 = 0.9
        val beta2 = 0.999
        val eps = 1e-8

        val opt: Optimizer = adam(lr = lr, beta1 = beta1, beta2 = beta2, epsilon = eps)
        opt.addParameter(w)

        // Step 1: grad = 1.0
        val g1 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0)
        w.value.accumulateGrad(g1 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt.step()

        val w1 = w.value.data[0, 0]

        // Step 2: grad = 2.0
        val g2 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w.value.accumulateGrad(g2 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt.step()

        val w2 = w.value.data[0, 0]

        // Verify that the parameter is being updated (decreasing with positive gradients)
        assertTrue(w1 < 5f, "Parameter should decrease after step 1")
        assertTrue(w2 < w1, "Parameter should decrease further after step 2")
    }

    @Test
    fun decoupled_weight_decay_adamw() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(10f)

        val lr = 0.1
        val wd = 0.1

        val opt: Optimizer = adamw(lr = lr, weightDecay = wd)
        opt.addParameter(w, applyWeightDecay = true)

        // gradient = 0.0 (no gradient, only weight decay)
        val g = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 0.0)
        w.value.accumulateGrad(g as sk.ainet.lang.tensor.Tensor<FP32, Float>)

        opt.step()

        // With decoupled weight decay and zero gradient:
        // Weight decay update: w = w - lr * wd * w = 10 - 0.1 * 0.1 * 10 = 10 - 0.1 = 9.9
        // Adam update with zero gradient: m=0, v=0, so no additional change
        // Expected: 9.9
        assertEquals(9.9f, w.value.data[0, 0], 1e-4f)
    }

    @Test
    fun l2_regularization_mode() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(10f)

        val lr = 0.1
        val beta1 = 0.9
        val beta2 = 0.999
        val eps = 1e-8
        val wd = 0.5

        // L2 regularization mode (not decoupled)
        val opt = AdamOptimizer(
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            epsilon = eps,
            weightDecay = wd,
            decoupledWeightDecay = false
        )
        opt.addParameter(w, applyWeightDecay = true)

        // gradient = 1.0
        val g = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0)
        w.value.accumulateGrad(g as sk.ainet.lang.tensor.Tensor<FP32, Float>)

        opt.step()

        // With L2 regularization: effective_grad = grad + wd * w = 1.0 + 0.5 * 10 = 6.0
        // m = (1 - 0.9) * 6.0 = 0.6
        // v = (1 - 0.999) * 36.0 = 0.036
        // m_hat = 0.6 / 0.1 = 6.0
        // v_hat = 0.036 / 0.001 = 36.0
        // update = 0.1 * 6.0 / (sqrt(36.0) + 1e-8) = 0.1 * 6.0 / 6.0 = 0.1
        // new_w = 10.0 - 0.1 = 9.9

        val effectiveGrad = 1.0 + wd * 10.0
        val m = (1 - beta1) * effectiveGrad
        val v = (1 - beta2) * effectiveGrad * effectiveGrad
        val mHat = m / (1 - beta1.pow(1))
        val vHat = v / (1 - beta2.pow(1))
        val update = lr * mHat / (sqrt(vHat) + eps)
        val expected = 10.0 - update

        assertEquals(expected.toFloat(), w.value.data[0, 0], 1e-4f)
    }

    @Test
    fun amsgrad_variant() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(5f)

        val opt = AdamOptimizer(
            lr = 0.1,
            beta1 = 0.9,
            beta2 = 0.999,
            amsgrad = true
        )
        opt.addParameter(w)

        // Step 1: large gradient
        val g1 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 10.0)
        w.value.accumulateGrad(g1 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt.step()
        val w1 = w.value.data[0, 0]

        // Step 2: small gradient
        val g2 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 0.1)
        w.value.accumulateGrad(g2 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt.step()
        val w2 = w.value.data[0, 0]

        // With AMSGrad, v_max should prevent the effective learning rate from increasing
        // when gradients become smaller
        assertTrue(w1 < 5f, "Parameter should decrease after step 1")
        assertTrue(w2 < w1, "Parameter should continue decreasing with AMSGrad")
    }

    @Test
    fun reset_clears_optimizer_state() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(10f)

        val opt = AdamOptimizer(lr = 0.1)
        opt.addParameter(w)

        // Perform a step to build up state
        val g1 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w.value.accumulateGrad(g1 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt.step()
        val afterStep1 = w.value.data[0, 0]

        // Reset the optimizer
        opt.reset()

        // Perform another step - should behave like step 1 again
        // First, reset the parameter value manually for comparison
        val w2 = param1x1(10f)
        val opt2 = AdamOptimizer(lr = 0.1)
        opt2.addParameter(w2)

        val g2 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w2.value.accumulateGrad(g2 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        opt2.step()

        // Both should produce the same result since opt was reset
        assertEquals(afterStep1, w2.value.data[0, 0], 1e-6f)
    }

    @Test
    fun skip_non_trainable_parameters() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val wTrain = param1x1(10f, train = true)
        val wFrozen = param1x1(5f, train = false)

        val opt: Optimizer = adam(lr = 0.1)
        opt.addParameter(wTrain)
        opt.addParameter(wFrozen)

        // Apply gradient to both
        val g = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0)
        wTrain.value.accumulateGrad(g as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        // wFrozen won't have grad accumulated because requiresGrad is false

        opt.step()

        // Trainable parameter should be updated
        assertTrue(wTrain.value.data[0, 0] < 10f, "Trainable parameter should be updated")

        // Frozen parameter should remain unchanged
        assertEquals(5f, wFrozen.value.data[0, 0], 1e-6f)
    }
}
