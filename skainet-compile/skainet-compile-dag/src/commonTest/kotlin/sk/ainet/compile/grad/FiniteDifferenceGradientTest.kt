package sk.ainet.compile.grad

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.tape.ExecutionTape

class FiniteDifferenceGradientTest {

    private fun fbuf(t: Tensor<FP32, Float>): FloatArray = (t.data as FloatArrayTensorData<FP32>).buffer

    private fun approxEqual(a: FloatArray, b: FloatArray, tol: Float = 1e-3f): Boolean {
        if (a.size != b.size) return false
        for (i in a.indices) {
            if (kotlin.math.abs(a[i] - b[i]) > tol) return false
        }
        return true
    }

    @Test
    fun add_gradient_matches_finite_difference() {
        val ctx = DefaultGraphExecutionContext.tape()
        // Ensure TRAIN phase to enable gradient recording
        val trainCtx = DefaultGraphExecutionContext(
            baseOps = ctx.baseOps,
            phase = Phase.TRAIN,
            tensorDataFactory = ctx.tensorDataFactory,
            hooks = ctx.hooks,
            memoryInfo = ctx.memoryInfo,
            executionStats = ctx.executionStats,
            createTapeFactory = { _ -> DefaultGradientTape(true) }
        )

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f)).withRequiresGrad()
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(0.5f, -1f, 2f))

        val pair1 = trainCtx.record {
            val out = a + b
            out.mean()
        }
        val tape = pair1.first
        val loss = pair1.second
        require(tape is DefaultGradientTape)

        // Backward: compute grads d(loss)/d(a)
        tape.computeGradients(targets = listOf(loss), sources = listOf(a))

        val analytic = fbuf(a.grad!!)

        // Finite differences
        val eps = 1e-3f
        val baseA0 = fbuf(a)
        val num = FloatArray(baseA0.size)
        for (i in baseA0.indices) {
            val plus = baseA0.copyOf(); plus[i] = plus[i] + eps
            val minus = baseA0.copyOf(); minus[i] = minus[i] - eps
            val aPlus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, plus)
            val aMinus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, minus)
            val lPlus = fbuf((aPlus + b).mean())[0]
            val lMinus = fbuf((aMinus + b).mean())[0]
            num[i] = (lPlus - lMinus) / (2 * eps)
        }

        assertTrue(approxEqual(analytic, num, tol = 5e-2f))
    }

    @Test
    fun relu_gradient_nonneg() {
        val ctx = DefaultGraphExecutionContext.tape()
        val trainCtx = DefaultGraphExecutionContext(
            baseOps = ctx.baseOps,
            phase = Phase.TRAIN,
            tensorDataFactory = ctx.tensorDataFactory,
            hooks = ctx.hooks,
            memoryInfo = ctx.memoryInfo,
            executionStats = ctx.executionStats,
            createTapeFactory = { _ -> DefaultGradientTape(true) }
        )

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(-1f, 0f, 2f)).withRequiresGrad()
        val pair2 = trainCtx.record { a.relu().sum() }
        val tape2 = pair2.first
        val loss2 = pair2.second
        require(tape2 is DefaultGradientTape)
        tape2.computeGradients(targets = listOf(loss2), sources = listOf(a))
        val g = fbuf(a.grad!!)
        println("[DEBUG_LOG] ReLU input: ${fbuf(a).joinToString()}")
        println("[DEBUG_LOG] ReLU output: ${fbuf(a.relu()).joinToString()}")
        println("[DEBUG_LOG] ReLU upstream: ${fbuf(trainCtx.record { a.relu().sum() }.second).joinToString()}")
        println("[DEBUG_LOG] ReLU grad: ${g.joinToString()}")
        // Allow 0 or 1 at boundary depending on implementation; ensure non-negative and g[2] approx 1
        assertTrue(g[0] == 0f && g[1] >= 0f && kotlin.math.abs(g[2] - 1f) < 1e-3)
    }
}
