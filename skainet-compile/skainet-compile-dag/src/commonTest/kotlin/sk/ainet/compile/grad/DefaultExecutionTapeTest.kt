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
import sk.ainet.lang.tensor.*
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.tensor.ops.AddOperation

class DefaultExecutionTapeTest {

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
    fun gelu_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()
        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(-1f, 0f, 1f)).withRequiresGrad()
        
        val pair = trainCtx.record { a.gelu().sum() }
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
            val lPlus = fbuf(aPlus.gelu().sum())[0]
            val lMinus = fbuf(aMinus.gelu().sum())[0]
            num[i] = (lPlus - lMinus) / (2 * eps)
        }

        assertTrue(approxEqual(analytic, num, tol = 5e-2f), "Analytic: ${analytic.joinToString()}, Numeric: ${num.joinToString()}")
    }

    @Test
    fun concat_gradient_matches_finite_difference() {
        val trainCtx = createTrainCtx()
        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(1f, 2f)).withRequiresGrad()
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(3f, 4f)).withRequiresGrad()

        val pair = trainCtx.record { trainCtx.ops.concat(listOf(a, b), 0).sum<FP32, Float>() }
        val tape = pair.first as DefaultGradientTape
        val loss = pair.second

        tape.computeGradients(targets = listOf(loss), sources = listOf(a, b))
        val analyticA = fbuf(a.grad!!)
        val analyticB = fbuf(b.grad!!)

        assertTrue(approxEqual(analyticA, floatArrayOf(1f, 1f)))
        assertTrue(approxEqual(analyticB, floatArrayOf(1f, 1f)))
    }

    @Test
    fun replay_works() {
        val trainCtx = createTrainCtx()
        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(1f, 2f))
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(3f, 4f))

        val tape = DefaultExecutionTape(trainCtx.session)
        tape.startRecording()
        // Manually record an operation
        val out = a + b
        // session.refOf is now handled by the trainCtx session which tape also uses
        tape.recordOperation(AddOperation<FP32, Float>(), listOf(a, b), listOf(out))
        tape.stopRecording()

        val replayed = tape.replay<FP32, Float>()
        assertTrue(replayed.size == 1)
        assertTrue(approxEqual(fbuf(replayed[0]), floatArrayOf(4f, 6f)))
    }

    @Test
    fun prune_works() {
        val trainCtx = createTrainCtx()
        val a = trainCtx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(1f))
        val b = trainCtx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(2f))

        val tape = DefaultExecutionTape()
        tape.startRecording()
        val out1 = a + b
        val out2 = out1 + b
        val out3 = a + a // dead code
        
        tape.session.refOf(a)
        tape.session.refOf(b)
        tape.session.refOf(out1)
        tape.session.refOf(out2)
        tape.session.refOf(out3)
        
        tape.recordOperation(AddOperation<FP32, Float>(), listOf(a, b), listOf(out1))
        tape.recordOperation(AddOperation<FP32, Float>(), listOf(out1, b), listOf(out2))
        tape.recordOperation(AddOperation<FP32, Float>(), listOf(a, a), listOf(out3))
        tape.stopRecording()

        assertTrue(tape.operations.size == 3)

        val out2Ref = tape.session.refOf(out2)
        val prunedTape = tape.prune(setOf(out2Ref.id)) as DefaultExecutionTape
        
        // Should only have out1 and out2 calculations
        assertTrue(prunedTape.operations.size == 2)
        assertTrue(prunedTape.operations[0].operation is AddOperation<*, *>)
        assertTrue(prunedTape.operations[1].operation is AddOperation<*, *>)
    }
}
