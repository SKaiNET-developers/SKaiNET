package sk.ainet.exec.autograd

import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.loss.Reduction
import sk.ainet.lang.nn.optim.adamw
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * System-level regression for #994. `gather()`'s primary documented use case — a batched
 * `[B,T]` token-embedding lookup — must not just avoid throwing during backward but actually
 * train: forward, backward, optimizer step, repeated, driving the loss down on a real (if
 * tiny) synthetic task. This is deliberately close to a minimal "bigram" next-token model —
 * the shape that surfaced the bug in the first place — rather than a synthetic op-level check.
 */
class EmbeddingTableTrainingE2ETest {

    private fun ctx(): DefaultGraphExecutionContext {
        val dataFactory = DenseTensorDataFactory()
        return DefaultGraphExecutionContext(
            baseOps = DefaultCpuOps(dataFactory),
            phase = Phase.TRAIN,
            tensorDataFactory = dataFactory,
            createTapeFactory = { _ -> DefaultGradientTape(true) },
        )
    }

    @Test
    fun batched_embedding_lookup_trains_without_throwing_and_reduces_loss() {
        val ctx = ctx()
        val rng = Random(42)
        val vocabSize = 6
        val batchSize = 4
        val blockSize = 3

        val tableData = ctx.tensorDataFactory.randn<FP32, Float>(Shape(vocabSize, vocabSize), FP32::class, 0f, 0.02f, rng)
        val table = ctx.fromData(tableData, FP32::class).withRequiresGrad(true)
        val param = ModuleParameter.WeightParameter("table", table, true)

        val optimizer = adamw(lr = 0.1)
        optimizer.addParameter(param)
        val lossFn = CrossEntropyLoss()

        // A tiny fixed pattern (next id = id+1 mod vocabSize) so there's something real to
        // learn — a bigram table can fit this exactly, unlike i.i.d. random targets.
        fun batch(): Pair<Tensor<Int32, Int>, Tensor<Int32, Int>> {
            val x = IntArray(batchSize * blockSize) { rng.nextInt(vocabSize) }
            val y = IntArray(x.size) { (x[it] + 1) % vocabSize }
            return ctx.fromIntArray<Int32, Int>(Shape(batchSize, blockSize), Int32::class, x) to
                ctx.fromIntArray<Int32, Int>(Shape(batchSize, blockSize), Int32::class, y)
        }

        @Suppress("UNCHECKED_CAST")
        fun step(): Float {
            val (x, y) = batch()
            ctx.startRecording()
            val loss = try {
                val logits = ctx.ops.gather(param.value, x as Tensor<DType, *>, dim = 0) // [B,T,vocab]
                val flatLogits = ctx.ops.reshape(logits, Shape(batchSize * blockSize, vocabSize))
                val flatTargets = ctx.ops.reshape(y, Shape(batchSize * blockSize))
                lossFn.forward(flatLogits, flatTargets, ctx, Reduction.MEAN)
            } finally {
                ctx.stopRecording()
            }
            ctx.backward(targets = listOf(loss), sources = listOf(param.value))
            optimizer.step()
            optimizer.zeroGrad()
            return loss.data.get()
        }

        val firstLoss = step()
        var lastLoss = firstLoss
        repeat(199) { lastLoss = step() }

        assertTrue(lastLoss < firstLoss, "loss should decrease over training: first=$firstLoss last=$lastLoss")
        val chanceLoss = ln(vocabSize.toFloat())
        assertTrue(
            lastLoss < 0.25f * chanceLoss,
            "a perfectly learnable deterministic next-token pattern should converge well below " +
                "chance loss ln(vocabSize)=$chanceLoss, got $lastLoss",
        )
    }
}
