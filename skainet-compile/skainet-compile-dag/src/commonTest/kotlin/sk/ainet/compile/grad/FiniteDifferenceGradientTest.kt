package sk.ainet.compile.grad

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.*
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

    private fun createTrainCtx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) }
        )
    }

    @Test
    fun add_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f)).withRequiresGrad()
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(0.5f, -1f, 2f))

        val pair1 = trainCtx.record {
            val out = a + b
            out.mean()
        }
        val tape = pair1.first as DefaultGradientTape
        val loss = pair1.second

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
        val trainCtx = createTrainCtx()

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(-1f, 0f, 2f)).withRequiresGrad()
        val pair2 = trainCtx.record { a.relu().sum() }
        val tape2 = pair2.first as DefaultGradientTape
        val loss2 = pair2.second
        
        tape2.computeGradients(targets = listOf(loss2), sources = listOf(a))
        val g = fbuf(a.grad!!)
        println("[DEBUG_LOG] ReLU input: ${fbuf(a).joinToString()}")
        println("[DEBUG_LOG] ReLU output: ${fbuf(a.relu()).joinToString()}")
        println("[DEBUG_LOG] ReLU grad: ${g.joinToString()}")
        // Allow 0 or 1 at boundary depending on implementation; ensure non-negative and g[2] approx 1
        assertTrue(g[0] == 0f && g[1] >= 0f && kotlin.math.abs(g[2] - 1f) < 1e-3)
    }

    @Test
    fun matmul_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()

        // y = A * B, loss = sum(y)
        // dL/dA = B^T
        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f)).withRequiresGrad()
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(0.5f, 0.1f, -0.2f, 0.8f))

        val pair = trainCtx.record {
            val out = a.matmul(b)
            out.sum()
        }
        val tape = pair.first as DefaultGradientTape
        val loss = pair.second
        tape.computeGradients(targets = listOf(loss), sources = listOf(a))

        val analytic = fbuf(a.grad!!)

        // Finite differences
        val eps = 1e-3f
        val baseA = fbuf(a)
        val num = FloatArray(baseA.size)
        for (i in baseA.indices) {
            val plus = baseA.copyOf(); plus[i] += eps
            val minus = baseA.copyOf(); minus[i] -= eps
            val aPlus = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, plus)
            val aMinus = trainCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, minus)
            val lPlus = fbuf(aPlus.matmul(b).sum())[0]
            val lMinus = fbuf(aMinus.matmul(b).sum())[0]
            num[i] = (lPlus - lMinus) / (2 * eps)
        }

        assertTrue(approxEqual(analytic, num, tol = 1e-2f))
    }

    @Test
    fun mul_div_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f)).withRequiresGrad()
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(2f, 4f, 8f)).withRequiresGrad()

        // Test Mul: out = a * b
        val pairMul = trainCtx.record { (a * b).sum() }
        val tapeMul = pairMul.first as DefaultGradientTape
        tapeMul.computeGradients(targets = listOf(pairMul.second), sources = listOf(a, b))
        
        val gradA_mul = fbuf(a.grad!!)
        val gradB_mul = fbuf(b.grad!!)
        
        assertTrue(approxEqual(gradA_mul, floatArrayOf(2f, 4f, 8f)))
        assertTrue(approxEqual(gradB_mul, floatArrayOf(1f, 2f, 3f)))
        
        a.zeroGrad()
        b.zeroGrad()

        // Test Div: out = a / b
        val pairDiv = trainCtx.record { (a / b).sum() }
        val tapeDiv = pairDiv.first as DefaultGradientTape
        tapeDiv.computeGradients(targets = listOf(pairDiv.second), sources = listOf(a, b))
        
        val gradA_div = fbuf(a.grad!!)
        val gradB_div = fbuf(b.grad!!)
        
        // d(a/b)/da = 1/b
        assertTrue(approxEqual(gradA_div, floatArrayOf(1/2f, 1/4f, 1/8f)))
        // d(a/b)/db = -a/b^2
        assertTrue(approxEqual(gradB_div, floatArrayOf(-1/4f, -2/16f, -3/64f)))
    }

    @Test
    fun sqrt_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 4f, 9f)).withRequiresGrad()
        val pair = trainCtx.record { a.sqrt().sum() }
        val tape = pair.first as DefaultGradientTape
        val loss = pair.second
        
        tape.computeGradients(targets = listOf(loss), sources = listOf(a))
        val analytic = fbuf(a.grad!!)

        // Finite differences
        val eps = 1e-3f
        val baseA = fbuf(a)
        val num = FloatArray(baseA.size)
        for (i in baseA.indices) {
            val plus = baseA.copyOf(); plus[i] += eps
            val minus = baseA.copyOf(); minus[i] -= eps
            val aPlus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, plus)
            val aMinus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, minus)
            val lPlus = fbuf(aPlus.sqrt().sum())[0]
            val lMinus = fbuf(aMinus.sqrt().sum())[0]
            num[i] = (lPlus - lMinus) / (2 * eps)
        }

        assertTrue(approxEqual(analytic, num, tol = 1e-2f))
    }

    @Test
    fun sigmoid_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()

        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(-1f, 0f, 1f)).withRequiresGrad()
        val pair = trainCtx.record { a.sigmoid().sum() }
        val tape = pair.first as DefaultGradientTape
        val loss = pair.second
        
        tape.computeGradients(targets = listOf(loss), sources = listOf(a))
        val analytic = fbuf(a.grad!!)

        // Finite differences
        val eps = 1e-3f
        val baseA = fbuf(a)
        val num = FloatArray(baseA.size)
        for (i in baseA.indices) {
            val plus = baseA.copyOf(); plus[i] += eps
            val minus = baseA.copyOf(); minus[i] -= eps
            val aPlus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, plus)
            val aMinus = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, minus)
            val lPlus = fbuf(aPlus.sigmoid().sum())[0]
            val lMinus = fbuf(aMinus.sigmoid().sum())[0]
            num[i] = (lPlus - lMinus) / (2 * eps)
        }

        assertTrue(approxEqual(analytic, num, tol = 1e-2f))
    }
}
