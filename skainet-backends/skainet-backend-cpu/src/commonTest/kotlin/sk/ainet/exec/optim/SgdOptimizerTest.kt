package sk.ainet.exec.optim

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.withRequiresGrad

class SgdOptimizerTest {

    private fun param1x1(v: Float, train: Boolean = true): ModuleParameter.WeightParameter<FP32, Float> {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val t = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, v).withRequiresGrad(train)
        @Suppress("UNCHECKED_CAST")
        return ModuleParameter.WeightParameter("w", t as sk.ainet.lang.tensor.Tensor<FP32, Float>, train)
    }

    @Test
    fun basic_sgd_step_updates_parameter() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(10f)

        // gradient = 2.0
        val g = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w.value.accumulateGrad(g as sk.ainet.lang.tensor.Tensor<FP32, Float>)

        val opt: Optimizer = sgd(lr = 0.1)
        opt.addParameter(w)

        opt.step()

        // expected: w = 10 - 0.1 * 2 = 9.8
        assertEquals(9.8f, w.value.data[0, 0], 1e-6f)
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

        val opt: Optimizer = sgd(lr = 0.01)
        opt.addParameter(w1)
        opt.addParameter(w2)

        opt.zeroGrad()

        assertNull(w1.value.grad)
        assertNull(w2.value.grad)
    }

    @Test
    fun momentum_and_weight_decay_behaviour() {
        val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val w = param1x1(5f)

        val opt: Optimizer = sgd(lr = 0.1, momentum = 0.9, weightDecay = 0.5)
        opt.addParameter(w, applyWeightDecay = true)

        // Step 1: grad = 1.0
        val g1 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 1.0)
        w.value.accumulateGrad(g1 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        // weight decay adds 0.5 * w (=2.5) to grad => effective g = 3.5
        // momentum buffer becomes v = 3.5
        // update = 0.1 * 3.5 = 0.35, new w = 5 - 0.35 = 4.65
        opt.step()
        assertEquals(4.65f, w.value.data[0, 0], 1e-4f)

        // Step 2: grad = 2.0
        val g2 = ctx.full<FP32, Float>(Shape(1, 1), FP32::class, 2.0)
        w.value.accumulateGrad(g2 as sk.ainet.lang.tensor.Tensor<FP32, Float>)
        // weight decay term now 0.5 * 4.65 = 2.325; effective g = 4.325
        // momentum: v = 0.9 * 3.5 + 4.325 = 7.475
        // update = 0.1 * 7.475 = 0.7475; new w = 4.65 - 0.7475 = 3.9025
        opt.step()
        assertEquals(3.9025f, w.value.data[0, 0], 1e-3f)
    }
}
