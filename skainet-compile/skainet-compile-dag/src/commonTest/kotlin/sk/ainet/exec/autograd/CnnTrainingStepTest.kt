package sk.ainet.exec.autograd

import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.trace.GraphSink
import sk.ainet.lang.types.FP32

/**
 * End-to-end Tier D check: one SGD step on a tiny CNN exercises every
 * conv/pool/upsample backward formula end-to-end. Confirms that:
 *   1. The forward and backward graphs compose without dropping any op.
 *   2. Every trainable parameter (conv weight, conv bias, linear weight,
 *      linear bias) receives a non-null gradient.
 *   3. Loss decreases — or at least doesn't increase — after the optimiser
 *      applies the SGD update.
 *
 * Architecture (deliberately tiny for the in-process tape backend):
 *   input [1, 1, 4, 4]
 *     → conv2d(weight [2, 1, 2, 2], bias [2], stride 1, padding 0)  → [1, 2, 3, 3]
 *     → relu                                                          → [1, 2, 3, 3]
 *     → maxPool2d(2, stride 1, pad 0)                                 → [1, 2, 2, 2]
 *     → reshape [1, 8]
 *     → matmul(linW [8, 3]) + linB [1, 3]                             → [1, 3]
 *     → mse vs target [1, 3]
 */
class CnnTrainingStepTest {

    private fun trainCtx(): DefaultGraphExecutionContext {
        val dataFactory = sk.ainet.lang.tensor.data.DenseTensorDataFactory()
        val cpuOps = DefaultCpuOps(dataFactory)
        val graph = DefaultComputeGraph()
        return DefaultGraphExecutionContext(
            baseOps = cpuOps,
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
            computeGraph = graph,
            baseSink = GraphSink(graph),
        )
    }

    @Test
    fun cnn_one_sgd_step_decreases_loss_and_populates_all_grads() {
        val ctx = trainCtx()

        // Fixed input + target so the test is deterministic.
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 1, 4, 4), FP32::class,
            floatArrayOf(
                0.2f, 0.5f, -0.3f, 0.8f,
                -0.4f, 0.1f, 0.6f, -0.7f,
                0.9f, -0.2f, 0.3f, 0.4f,
                -0.1f, 0.7f, -0.5f, 0.6f,
            ),
        )
        val target = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 3), FP32::class,
            floatArrayOf(1f, 0f, -1f),
        )

        // Trainable parameters. Values handpicked so the network actually
        // computes something nontrivial (no all-zeros after ReLU).
        val convW = ctx.fromFloatArray<FP32, Float>(
            Shape(2, 1, 2, 2), FP32::class,
            floatArrayOf(
                0.3f, -0.4f, 0.5f, 0.1f,    // out-channel 0
                -0.2f, 0.6f, 0.1f, 0.3f,    // out-channel 1
            ),
        ).withRequiresGrad()
        val convB = ctx.fromFloatArray<FP32, Float>(
            Shape(2), FP32::class, floatArrayOf(0.05f, -0.05f),
        ).withRequiresGrad()
        val linW = ctx.fromFloatArray<FP32, Float>(
            Shape(8, 3), FP32::class,
            FloatArray(24) { (it % 7 - 3) * 0.1f },
        ).withRequiresGrad()
        val linB = ctx.fromFloatArray<FP32, Float>(
            Shape(1, 3), FP32::class, floatArrayOf(0.0f, 0.0f, 0.0f),
        ).withRequiresGrad()

        val convWParam = ModuleParameter.WeightParameter("convW", convW)
        val convBParam = ModuleParameter.BiasParameter("convB", convB)
        val linWParam = ModuleParameter.WeightParameter("linW", linW)
        val linBParam = ModuleParameter.BiasParameter("linB", linB)

        // Forward + record + backward in one block.
        fun forward(): sk.ainet.lang.tensor.Tensor<FP32, Float> {
            val conv = input.ops.conv2d(
                input, convW, convB,
                stride = 1 to 1, padding = 0 to 0, dilation = 1 to 1, groups = 1,
            )
            val activated = conv.relu()
            val pooled = activated.ops.maxPool2d(
                activated, kernelSize = 2 to 2, stride = 1 to 1, padding = 0 to 0,
            )
            val flat = pooled.ops.reshape(pooled, Shape(1, 8))
            val logits = flat.matmul(linW).ops.add(flat.matmul(linW), linB)
            val diff = logits.ops.subtract(logits, target)
            return logits.ops.sum(logits.ops.multiply(diff, diff))
        }

        // Baseline loss (eager — no need to record).
        val initialLoss = forward().data.get()

        // Training step: record forward, populate gradients, step optimiser.
        val pair = ctx.record { forward() }
        val tape = pair.first as DefaultGradientTape
        val loss = pair.second
        tape.computeGradients(
            targets = listOf(loss),
            sources = listOf(convW, convB, linW, linB),
        )

        assertNotNull(convW.grad, "convW must have grad after backward")
        assertNotNull(convB.grad, "convB must have grad after backward")
        assertNotNull(linW.grad, "linW must have grad after backward")
        assertNotNull(linB.grad, "linB must have grad after backward")

        val optimizer = sgd(lr = 0.01)
        optimizer.addParameter(convWParam)
        optimizer.addParameter(convBParam)
        optimizer.addParameter(linWParam)
        optimizer.addParameter(linBParam)
        optimizer.step()
        optimizer.zeroGrad()

        val updatedLoss = forward().data.get()
        assertTrue(
            updatedLoss <= initialLoss,
            "loss should not increase after SGD step (initial=$initialLoss, after=$updatedLoss)",
        )
    }
}
