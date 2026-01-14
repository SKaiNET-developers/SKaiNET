package sk.ainet.apps.sine

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.trainStep
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32
import sk.ainet.context.ExecutionContext
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Comprehensive tests for verifying each component of the training pipeline
 * using exact data from the sine approximation example.
 */
class SineTrainingComponentsTest {

    private fun createTrainCtx(): DefaultGraphExecutionContext {
        val baseCtx = DirectCpuExecutionContext()
        return DefaultGraphExecutionContext(
            baseOps = baseCtx.ops,
            phase = Phase.TRAIN,
            createTapeFactory = { _ -> DefaultGradientTape(true) }
        )
    }

    private fun createBaseCtx(): DirectCpuExecutionContext = DirectCpuExecutionContext()

    // ============================================================
    // TEST 1: MSE Loss Forward Computation
    // ============================================================
    @Test
    fun test_MSELoss_forward_computation() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Simple test case: pred = [1, 2, 3], target = [1, 2, 3] -> MSE = 0
        val pred = baseCtx.fromFloatArray<FP32, Float>(Shape(3, 1), FP32::class, floatArrayOf(1f, 2f, 3f))
        val target = baseCtx.fromFloatArray<FP32, Float>(Shape(3, 1), FP32::class, floatArrayOf(1f, 2f, 3f))

        val loss = MSELoss()
        val result = loss.forward(pred, target, ctx)
        val lossValue = result.data.get() as Float

        assertEquals(0f, lossValue, 1e-6f, "MSE should be 0 when pred == target")
    }

    @Test
    fun test_MSELoss_forward_nonzero() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // pred = [0, 0], target = [1, 1] -> diff = [-1, -1], squared = [1, 1], mean = 1
        val pred = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        val target = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 1f))

        val loss = MSELoss()
        val result = loss.forward(pred, target, ctx)
        val lossValue = result.data.get() as Float

        assertEquals(1f, lossValue, 1e-6f, "MSE of [0,0] vs [1,1] should be 1")
    }

    @Test
    fun test_MSELoss_forward_with_sine_data() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Use exact values from sine approximation
        val batchSize = 4
        val xValues = FloatArray(batchSize) { i ->
            (i.toFloat() / (batchSize - 1)) * (PI.toFloat() / 2f)
        }
        val yValues = FloatArray(batchSize) { i -> sin(xValues[i].toDouble()).toFloat() }

        // Simulate predictions that are all zeros
        val pred = baseCtx.fromFloatArray<FP32, Float>(Shape(batchSize, 1), FP32::class, FloatArray(batchSize) { 0f })
        val target = baseCtx.fromFloatArray<FP32, Float>(Shape(batchSize, 1), FP32::class, yValues)

        val loss = MSELoss()
        val result = loss.forward(pred, target, ctx)
        val lossValue = result.data.get() as Float

        // Expected: mean of squared sine values
        val expectedLoss = yValues.map { it * it }.average().toFloat()
        assertEquals(expectedLoss, lossValue, 1e-5f, "MSE should match expected value")
    }

    // ============================================================
    // TEST 2: Mean Operation Backward
    // ============================================================
    @Test
    fun test_mean_backward_gradient_shape() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Create a tensor that requires grad
        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(4, 1), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        x.gradState.requiresGrad = true

        ctx.startRecording()
        val result = ctx.ops.mean(x, null) // Full reduction
        ctx.stopRecording()

        // Run backward
        ctx.backward(listOf(result), listOf(x))

        // Check gradient exists and has correct shape
        val grad = x.grad
        assertNotNull(grad, "Gradient should be computed")
        assertEquals(x.shape, grad.shape, "Gradient should have same shape as input")

        // Gradient of mean is 1/N for each element
        val expectedGrad = 1f / 4f
        for (i in 0 until 4) {
            val actualGrad = grad.data[i, 0]
            assertEquals(expectedGrad, actualGrad, 1e-6f, "Gradient at index $i should be 1/N")
        }
    }

    // ============================================================
    // TEST 3: Multiply Backward (for squared = diff * diff)
    // ============================================================
    @Test
    fun test_multiply_backward_same_tensor() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // When we do x * x, gradient should be 2 * x * upstream
        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(3f, 4f))
        x.gradState.requiresGrad = true

        ctx.startRecording()
        val squared = ctx.ops.multiply(x, x)
        ctx.stopRecording()

        ctx.backward(listOf(squared), listOf(x))

        val grad = x.grad
        assertNotNull(grad, "Gradient should be computed")

        // d(x*x)/dx = 2*x, with upstream = 1
        // grad[0] should be 2 * 3 = 6
        // grad[1] should be 2 * 4 = 8
        assertEquals(6f, grad.data[0, 0], 1e-5f, "Gradient at 0 should be 2*x[0]=6")
        assertEquals(8f, grad.data[1, 0], 1e-5f, "Gradient at 1 should be 2*x[1]=8")
    }

    // ============================================================
    // TEST 4: Subtract Backward
    // ============================================================
    @Test
    fun test_subtract_backward() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        val a = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(5f, 6f))
        val b = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        a.gradState.requiresGrad = true
        b.gradState.requiresGrad = true

        ctx.startRecording()
        val diff = ctx.ops.subtract(a, b)
        ctx.stopRecording()

        ctx.backward(listOf(diff), listOf(a, b))

        val gradA = a.grad
        val gradB = b.grad
        assertNotNull(gradA, "Gradient for a should be computed")
        assertNotNull(gradB, "Gradient for b should be computed")

        // d(a-b)/da = 1, d(a-b)/db = -1
        assertEquals(1f, gradA.data[0, 0], 1e-6f)
        assertEquals(1f, gradA.data[1, 0], 1e-6f)
        assertEquals(-1f, gradB.data[0, 0], 1e-6f)
        assertEquals(-1f, gradB.data[1, 0], 1e-6f)
    }

    // ============================================================
    // TEST 5: ReLU Backward
    // ============================================================
    @Test
    fun test_relu_backward() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // x = [-1, 0, 1, 2]
        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(4, 1), FP32::class, floatArrayOf(-1f, 0f, 1f, 2f))
        x.gradState.requiresGrad = true

        ctx.startRecording()
        val y = ctx.ops.relu(x)
        ctx.stopRecording()

        ctx.backward(listOf(y), listOf(x))

        val grad = x.grad
        assertNotNull(grad, "Gradient should be computed")

        // ReLU gradient: 0 for x <= 0, 1 for x > 0
        assertEquals(0f, grad.data[0, 0], 1e-6f, "Gradient should be 0 for negative input")
        assertEquals(0f, grad.data[1, 0], 1e-6f, "Gradient should be 0 for zero input")
        assertEquals(1f, grad.data[2, 0], 1e-6f, "Gradient should be 1 for positive input")
        assertEquals(1f, grad.data[3, 0], 1e-6f, "Gradient should be 1 for positive input")
    }

    // ============================================================
    // TEST 6: Matmul Backward
    // ============================================================
    @Test
    fun test_matmul_backward() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // A: [2, 3], B: [3, 2]
        val a = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class,
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = baseCtx.fromFloatArray<FP32, Float>(Shape(3, 2), FP32::class,
            floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f))
        a.gradState.requiresGrad = true
        b.gradState.requiresGrad = true

        ctx.startRecording()
        val c = ctx.ops.matmul(a, b)
        ctx.stopRecording()

        ctx.backward(listOf(c), listOf(a, b))

        val gradA = a.grad
        val gradB = b.grad
        assertNotNull(gradA, "Gradient for A should be computed")
        assertNotNull(gradB, "Gradient for B should be computed")

        // Verify shapes
        assertEquals(a.shape, gradA.shape, "Gradient for A should have same shape as A")
        assertEquals(b.shape, gradB.shape, "Gradient for B should have same shape as B")

        // d(A@B)/dA = upstream @ B^T
        // d(A@B)/dB = A^T @ upstream
        // With upstream = ones([2,2])
        // gradA = ones([2,2]) @ B^T = ones([2,2]) @ [[1,0,1],[0,1,1]]
        //       = [[1+0+1, 0+1+1], [1+0+1, 0+1+1]] = [[2,2], [2,2]] -- wrong, recalculating
        // Actually B = [[1,0],[0,1],[1,1]] so B^T = [[1,0,1],[0,1,1]]
        // gradA[i,j] = sum_k(upstream[i,k] * B^T[k,j]) = sum_k(1 * B[j,k])
        // gradA[0,0] = B[0,0] + B[0,1] = 1 + 0 = 1
        // gradA[0,1] = B[1,0] + B[1,1] = 0 + 1 = 1
        // gradA[0,2] = B[2,0] + B[2,1] = 1 + 1 = 2
        // etc.

        println("gradA shape: ${gradA.shape}")
        println("gradA: ${(0 until 6).map { gradA.data.get(it / 3, it % 3) }}")
        println("gradB shape: ${gradB.shape}")
        println("gradB: ${(0 until 6).map { gradB.data.get(it / 2, it % 2) }}")
    }

    // ============================================================
    // TEST 7: Linear Layer Backward
    // ============================================================
    @Test
    fun test_linear_layer_backward() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Simple linear: y = x @ W^T + b
        // x: [2, 3], W: [4, 3], b: [4] -> y: [2, 4]
        val w = baseCtx.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class,
            FloatArray(12) { 0.1f })
        val b = baseCtx.fromFloatArray<FP32, Float>(Shape(4), FP32::class,
            FloatArray(4) { 0.0f })

        val linear = Linear<FP32, Float>(
            inFeatures = 3,
            outFeatures = 4,
            name = "test_linear",
            initWeights = w,
            initBias = b,
            trainable = true
        )

        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class,
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))

        ctx.startRecording()
        val y = linear.forward(x, ctx)
        ctx.stopRecording()

        val params = linear.trainableParameters()
        ctx.backward(listOf(y), params.map { it.value })

        // Check gradients exist
        for (p in params) {
            assertNotNull(p.value.grad, "Gradient for ${p.name} should be computed")
            println("${p.name} grad shape: ${p.value.grad?.shape}")
        }
    }

    // ============================================================
    // TEST 8: SGD Optimizer Step
    // ============================================================
    @Test
    fun test_sgd_optimizer_step() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Create a simple parameter
        val w = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class,
            floatArrayOf(1f, 2f, 3f, 4f))
        w.gradState.requiresGrad = true

        // Manually set gradient
        val grad = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class,
            floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f))
        w.accumulateGrad(grad)

        val param = ModuleParameter.WeightParameter<FP32, Float>("w", w, true)
        val optimizer = sgd(lr = 0.1)
        optimizer.addParameter(param)

        // Store initial values
        val initialW00 = param.value.data[0, 0]
        val initialW01 = param.value.data[0, 1]

        // Step
        optimizer.step()

        // w_new = w_old - lr * grad
        // w[0,0] = 1 - 0.1 * 0.1 = 0.99
        // w[0,1] = 2 - 0.1 * 0.2 = 1.98
        assertEquals(0.99f, param.value.data[0, 0], 1e-5f, "Weight [0,0] should be updated")
        assertEquals(1.98f, param.value.data[0, 1], 1e-5f, "Weight [0,1] should be updated")

        // Zero grad
        optimizer.zeroGrad()
        assertTrue(param.value.grad == null, "Gradient should be zeroed after zeroGrad()")
    }

    // ============================================================
    // TEST 9: Full MSE Loss Backward Chain
    // ============================================================
    @Test
    fun test_mse_loss_backward_chain() {
        val ctx = createTrainCtx()

        // KEY FIX: Create tensors from training context so they use recording ops
        // When created from baseCtx, tensors have baseCtx.ops which doesn't record!
        val pred = ctx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target = ctx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred.gradState.requiresGrad = true

        // pred = [1, 2], target = [0, 0]
        // diff = [1, 2], squared = [1, 4], mean = 2.5
        // d(mean)/d(squared) = [0.5, 0.5]
        // d(squared)/d(diff) = 2 * diff = [2, 4]
        // d(diff)/d(pred) = 1
        // Total: d(loss)/d(pred) = 2 * diff * 0.5 = diff = [1, 2]

        ctx.startRecording()
        val loss = MSELoss().forward(pred, target, ctx)
        ctx.stopRecording()

        println("Loss value: ${loss.data.get()}")
        println("Loss shape: ${loss.shape}")
        println("Pred requiresGrad: ${pred.requiresGrad}")

        ctx.backward(listOf(loss), listOf(pred))

        val grad = pred.grad
        println("Gradient: $grad")
        println("Gradient shape: ${grad?.shape}")
        if (grad != null) {
            println("Gradient values: ${(0 until 2).map { grad.data[it, 0] }}")
        }
        assertNotNull(grad, "Gradient should be computed")

        // Expected gradient: 2 * (pred - target) / N = 2 * [1, 2] / 2 = [1, 2]
        assertEquals(1f, grad.data[0, 0], 1e-5f, "Gradient at [0] should be 1")
        assertEquals(2f, grad.data[1, 0], 1e-5f, "Gradient at [1] should be 2")
    }

    // ============================================================
    // TEST 9c: Verify tensor ops binding is the root cause
    // ============================================================
    @Test
    fun test_tensor_ops_binding_root_cause() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // This test demonstrates the root cause: tensors created from different contexts
        // have different ops bound to them, affecting gradient recording

        println("=== Testing with baseCtx-created tensors (BREAKS gradient recording) ===")
        val pred1 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target1 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred1.gradState.requiresGrad = true
        println("pred1.ops class: ${pred1.ops::class.simpleName}")

        ctx.startRecording()
        // When using pred - target, it uses pred.ops which is baseCtx's ops (no recording)
        val diff1 = pred1 - target1
        ctx.stopRecording()
        ctx.backward(listOf(diff1), listOf(pred1))
        println("Gradient using pred.ops (baseCtx): ${pred1.grad?.let { (0 until 2).map { i -> it.data[i, 0] } } ?: "null"}")

        println("\n=== Testing with ctx-created tensors (WORKS correctly) ===")
        val pred2 = ctx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target2 = ctx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred2.gradState.requiresGrad = true
        println("pred2.ops class: ${pred2.ops::class.simpleName}")

        ctx.startRecording()
        // Now using pred.ops which is ctx's ops (records correctly)
        val diff2 = pred2 - target2
        ctx.stopRecording()
        ctx.backward(listOf(diff2), listOf(pred2))
        println("Gradient using pred.ops (ctx): ${pred2.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")

        // The ctx-created tensor should have gradient
        assertNotNull(pred2.grad, "Gradient should be computed for ctx-created tensor")
        assertEquals(1f, pred2.grad!!.data[0, 0], 1e-6f)
    }

    // ============================================================
    // TEST 9b: Step-by-step MSE backward debugging
    // ============================================================
    @Test
    fun test_mse_backward_step_by_step() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        // Step 1: Test subtract backward
        println("=== Step 1: Subtract Backward ===")
        val pred1 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target1 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred1.gradState.requiresGrad = true

        ctx.startRecording()
        val diff = ctx.ops.subtract(pred1, target1)
        ctx.stopRecording()
        ctx.backward(listOf(diff), listOf(pred1))
        println("Diff grad on pred: ${pred1.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")
        pred1.zeroGrad()

        // Step 2: Test multiply backward (x * x)
        println("\n=== Step 2: Multiply Backward (x*x) ===")
        val x2 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        x2.gradState.requiresGrad = true

        ctx.startRecording()
        val squared = ctx.ops.multiply(x2, x2)
        ctx.stopRecording()
        ctx.backward(listOf(squared), listOf(x2))
        println("Squared grad on x: ${x2.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")
        x2.zeroGrad()

        // Step 3: Test mean backward
        println("\n=== Step 3: Mean Backward ===")
        val x3 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 4f))
        x3.gradState.requiresGrad = true

        ctx.startRecording()
        val meanVal = ctx.ops.mean(x3, null)
        ctx.stopRecording()
        println("Mean value: ${meanVal.data.get()}")
        ctx.backward(listOf(meanVal), listOf(x3))
        println("Mean grad on x: ${x3.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")
        x3.zeroGrad()

        // Step 4: Test subtract -> multiply chain
        println("\n=== Step 4: Subtract -> Multiply Chain ===")
        val pred4 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target4 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred4.gradState.requiresGrad = true

        ctx.startRecording()
        val diff4 = ctx.ops.subtract(pred4, target4)
        val squared4 = ctx.ops.multiply(diff4, diff4)
        ctx.stopRecording()
        ctx.backward(listOf(squared4), listOf(pred4))
        println("Sub->Mul grad on pred: ${pred4.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")
        pred4.zeroGrad()

        // Step 5: Full chain subtract -> multiply -> mean
        println("\n=== Step 5: Full Chain (Sub -> Mul -> Mean) ===")
        val pred5 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(1f, 2f))
        val target5 = baseCtx.fromFloatArray<FP32, Float>(Shape(2, 1), FP32::class, floatArrayOf(0f, 0f))
        pred5.gradState.requiresGrad = true

        ctx.startRecording()
        val diff5 = ctx.ops.subtract(pred5, target5)
        val squared5 = ctx.ops.multiply(diff5, diff5)
        val loss5 = ctx.ops.mean(squared5, null)
        ctx.stopRecording()
        println("Loss value: ${loss5.data.get()}")
        ctx.backward(listOf(loss5), listOf(pred5))
        println("Full chain grad on pred: ${pred5.grad?.let { (0 until 2).map { i -> it.data[i, 0] } }}")

        // Verify gradients
        val grad5 = pred5.grad
        assertNotNull(grad5, "Gradient should be computed for full chain")
        // Expected: 2 * (pred - target) / N = 2 * [1, 2] / 2 = [1, 2]
        assertEquals(1f, grad5.data[0, 0], 1e-5f, "Gradient at [0] should be 1")
        assertEquals(2f, grad5.data[1, 0], 1e-5f, "Gradient at [1] should be 2")
    }

    // ============================================================
    // TEST 10: End-to-End Single Training Step
    // ============================================================
    @Test
    fun test_single_training_step_loss_decreases() {
        val ctx = createTrainCtx()

        // Simple model: y = w * x (single weight)
        val wInitial = 10f
        // Use ctx directly to ensure registration
        val wTensor = ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(wInitial))
        wTensor.gradState.requiresGrad = true
        val w = ModuleParameter.WeightParameter<FP32, Float>("w", wTensor, true)

        val model = object : Module<FP32, Float>() {
            override val name: String = "simple"
            override val modules: List<Module<FP32, Float>> = emptyList()
            override val params: List<ModuleParameter<FP32, Float>> = listOf(w)
            override fun forward(input: Tensor<FP32, Float>, ctx: ExecutionContext): Tensor<FP32, Float> {
                return ctx.ops.matmul(input, w.value) as Tensor<FP32, Float>
            }
        }

        val loss = MSELoss()
        val optimizer = sgd(lr = 0.1)
        optimizer.addParameter(w)

        // Input x=1, target y=5
        // Initial pred = 10 * 1 = 10
        // Initial loss = (10 - 5)^2 = 25
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(1.0f))
        val y = ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(5.0f))

        val initialLoss = trainStep(model, loss, optimizer, ctx, x, y)
        val initialLossValue = initialLoss.data.get() as Float

        println("Initial loss: $initialLossValue, Initial weight: $wInitial")
        println("Weight after step: ${w.value.data[0, 0]}")

        // Run another step
        val secondLoss = trainStep(model, loss, optimizer, ctx, x, y)
        val secondLossValue = secondLoss.data.get() as Float

        println("Second loss: $secondLossValue, Weight: ${w.value.data[0, 0]}")

        assertTrue(secondLossValue < initialLossValue,
            "Loss should decrease after training step. Initial: $initialLossValue, After: $secondLossValue")
        assertTrue(w.value.data[0, 0] < wInitial,
            "Weight should move towards target (decrease from 10 towards 5)")
    }

    // ============================================================
    // TEST 11: Multi-Layer Network Training Step
    // ============================================================
    @Test
    fun test_multilayer_training_step() {
        val ctx = createTrainCtx()

        // Create a simple 2-layer network: input -> linear(4) -> relu -> linear(1)
        val model = sequential<FP32, Float>(ctx) {
            input(1)
            dense(4) {
                weights { randn(std = 0.5f) }
            }
            activation { it.relu() }
            dense(1) {
                weights { randn(std = 0.5f) }
            }
        }

        val runner = training<FP32, Float> {
            model { model }
            loss { MSELoss() }
            optimizer {
                sgd(lr = 0.01).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        // Simple data: x=0.5, y=sin(0.5)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(0.5f))
        val y = ctx.fromFloatArray<FP32, Float>(Shape(1, 1), FP32::class, floatArrayOf(sin(0.5).toFloat()))

        // Helper to get first element regardless of tensor shape
        fun getFirstElement(t: Tensor<*, *>): Float {
            return if (t.rank == 1) t.data[0] as Float else t.data[0, 0] as Float
        }

        // Store initial weights for comparison
        val initialParams = model.trainableParameters().map {
            it.name to getFirstElement(it.value)
        }.toMap()

        // Run training step
        val loss1 = runner.step(ctx, x, y)
        val lossValue1 = loss1.data.get() as Float

        // Verify parameters changed
        var anyChanged = false
        model.trainableParameters().forEach { p ->
            val initial = initialParams[p.name]
            val current = getFirstElement(p.value)
            if (initial != null && abs(current - initial) > 1e-8f) {
                anyChanged = true
                println("${p.name}: $initial -> $current (changed by ${current - initial})")
            }
        }

        assertTrue(anyChanged, "At least one parameter should change after training step")

        // Run multiple steps and verify loss decreases
        var prevLoss = lossValue1
        repeat(10) { epoch ->
            val loss = runner.step(ctx, x, y)
            val currentLoss = loss.data.get() as Float
            println("Epoch ${epoch + 1}: loss = $currentLoss")
            prevLoss = currentLoss
        }
    }

    // ============================================================
    // TEST 12: Sine Approximation Exact Data Test
    // ============================================================
    @Test
    fun test_sine_approximation_with_exact_data() {
        val ctx = createTrainCtx()

        val batchSize = 8
        val xValues = FloatArray(batchSize) { i ->
            (i.toFloat() / (batchSize - 1)) * (PI.toFloat() / 2f)
        }
        val yValues = FloatArray(batchSize) { i -> sin(xValues[i].toDouble()).toFloat() }

        println("Training data:")
        xValues.zip(yValues.toList()).forEach { (x, y) ->
            println("  x = $x, sin(x) = $y")
        }

        // Create model matching SineApproxCli
        val model = sequential<FP32, Float>(ctx) {
            input(1)
            dense(16) {
                weights { randn(std = 0.5f) }
            }
            activation { it.relu() }
            dense(16) {
                weights { randn(std = 0.5f) }
            }
            activation { it.relu() }
            dense(1) {
                weights { randn(std = 0.5f) }
            }
        }

        val runner = training<FP32, Float> {
            model { model }
            loss { MSELoss() }
            optimizer {
                sgd(lr = 0.05).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        val inputs = ctx.fromFloatArray<FP32, Float>(Shape(batchSize, 1), FP32::class, xValues)
        val targets = ctx.fromFloatArray<FP32, Float>(Shape(batchSize, 1), FP32::class, yValues)

        // Track initial loss
        val initialLoss = runner.step(ctx, inputs, targets)
        val initialLossValue = initialLoss.data.get() as Float
        println("Initial loss: $initialLossValue")

        // Train for some epochs
        var lastLoss = initialLossValue
        repeat(50) { epoch ->
            val loss = runner.step(ctx, inputs, targets)
            lastLoss = loss.data.get() as Float
            if ((epoch + 1) % 10 == 0) {
                println("Epoch ${epoch + 2}: loss = $lastLoss")
            }
        }

        println("Final loss: $lastLoss (initial was $initialLossValue)")
        println("Loss decreased by: ${initialLossValue - lastLoss}")

        // The loss should decrease
        assertTrue(lastLoss < initialLossValue,
            "Loss should decrease over training. Initial: $initialLossValue, Final: $lastLoss")
    }

    // ============================================================
    // TEST 13: Gradient Numerical Check for Simple Case
    // ============================================================
    @Test
    fun test_gradient_numerical_check() {
        val ctx = createTrainCtx()
        val baseCtx = createBaseCtx()

        val eps = 1e-4f

        // Test: f(x) = x^2, df/dx = 2x
        val x0 = 3.0f
        val x = baseCtx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(x0))
        x.gradState.requiresGrad = true

        ctx.startRecording()
        val y = ctx.ops.multiply(x, x)
        ctx.stopRecording()

        ctx.backward(listOf(y), listOf(x))

        val analyticGrad = x.grad?.data?.get(0) as Float

        // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        val fPlus = (x0 + eps) * (x0 + eps)
        val fMinus = (x0 - eps) * (x0 - eps)
        val numericalGrad = (fPlus - fMinus) / (2 * eps)

        println("Analytic gradient: $analyticGrad")
        println("Numerical gradient: $numericalGrad")
        println("Expected (2*x): ${2 * x0}")

        assertEquals(analyticGrad, numericalGrad, 1e-2f,
            "Analytic and numerical gradients should match")
        assertEquals(analyticGrad, 2 * x0, 1e-3f,
            "Gradient should be 2*x")
    }
}
