package sk.ainet.lang.nn

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.dsl.training
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.optim.sgd
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.tanh
import sk.ainet.lang.types.FP32
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Faithful port of Karpathy's micrograd demo.ipynb to SKaiNET — without visualisations.
 *
 * Mirrors the original demo end-to-end:
 *   1. Generate the two-moons dataset (interleaving half-circles) with Gaussian noise.
 *   2. Preprocess features with z-score normalisation.
 *   3. Define the MLP `[2, 16, 16, 1]` with `tanh` activations using SKaiNET's `sequential { ... }` DSL —
 *      this is the consumer of issue #630's first-class `tanh` primitive.
 *   4. Train with SKaiNET's `training { ... }` DSL using SGD + MSE-against-±1 (functionally
 *      equivalent to micrograd's max-margin loss when the output passes through `tanh`).
 *   5. Numeric evaluation: classification accuracy on a held-out split must exceed a threshold.
 *
 * Everything is deterministic (seeded `Random`) so the accuracy assertion is stable across runs.
 */
class MicrogradMoonsDemoTest {

    @Test
    fun micrograd_moons_demo_trains_and_classifies() {
        val seed = 1337
        val nSamples = 100
        val noise = 0.1f
        val epochs = 200
        val lr = 0.1

        // ------------------------------------------------------------------
        // 1. Data: generate moons + 80/20 train/eval split
        // ------------------------------------------------------------------
        val (rawX, rawY) = makeMoons(nSamples = nSamples, noise = noise, seed = seed)
        val splitIdx = (nSamples * 0.8f).toInt()
        val trainX = rawX.copyOfRange(0, splitIdx * 2)
        val trainY = rawY.copyOfRange(0, splitIdx)
        val evalX = rawX.copyOfRange(splitIdx * 2, rawX.size)
        val evalY = rawY.copyOfRange(splitIdx, rawY.size)

        // ------------------------------------------------------------------
        // 2. Preprocessing pipeline: z-score normalisation, fit on train, apply to eval
        // ------------------------------------------------------------------
        val (mean, std) = featurewiseMeanStd(trainX, features = 2)
        zScoreInPlace(trainX, mean, std, features = 2)
        zScoreInPlace(evalX, mean, std, features = 2)

        // Sanity check: train set should now be ~zero mean, unit std per feature.
        val (mTrain, sTrain) = featurewiseMeanStd(trainX, features = 2)
        for (f in 0 until 2) {
            assertTrue(kotlin.math.abs(mTrain[f]) < 1e-4f, "feature $f mean ~ 0 after z-score, got ${mTrain[f]}")
            assertTrue(kotlin.math.abs(sTrain[f] - 1f) < 1e-4f, "feature $f std ~ 1 after z-score, got ${sTrain[f]}")
        }

        // ------------------------------------------------------------------
        // 3. Model: MLP [2, 16, 16, 1] with tanh activations — the Karpathy net.
        //    Tanh on the final layer keeps outputs in (-1, +1) for the ±1 targets.
        // ------------------------------------------------------------------
        val baseCtx = DirectCpuExecutionContext()
        val trainCtx = DefaultGraphExecutionContext(
            baseOps = baseCtx.ops,
            phase = Phase.TRAIN,
            createTapeFactory = { _ -> DefaultGradientTape() }
        )
        val initRng = Random(seed)
        val model = sequential<FP32, Float>(trainCtx) {
            input(2)
            dense(16) { weights { randn(std = 0.3f, random = initRng) } }
            activation { it.tanh() }
            dense(16) { weights { randn(std = 0.3f, random = initRng) } }
            activation { it.tanh() }
            dense(1) { weights { randn(std = 0.3f, random = initRng) } }
            activation { it.tanh() }
        }

        // ------------------------------------------------------------------
        // 4. Training: SKaiNET's training DSL with SGD + MSE-against-±1
        // ------------------------------------------------------------------
        val xTrain = baseCtx.fromFloatArray<FP32, Float>(Shape(splitIdx, 2), FP32::class, trainX)
        val yTrain = baseCtx.fromFloatArray<FP32, Float>(Shape(splitIdx, 1), FP32::class, trainY)

        val runner = training<FP32, Float> {
            model { model }
            loss { MSELoss() }
            optimizer {
                sgd(lr = lr).apply {
                    model.trainableParameters().forEach { addParameter(it) }
                }
            }
        }

        var firstLoss = 0f
        var lastLoss = 0f
        repeat(epochs) { epoch ->
            val lossTensor = runner.step(trainCtx, xTrain, yTrain)
            val l = lossTensor.data.get() as Float
            if (epoch == 0) firstLoss = l
            lastLoss = l
            if ((epoch + 1) % 20 == 0) {
                println("[DEBUG_LOG] Epoch ${epoch + 1}/$epochs, loss=$l")
            }
        }
        println("[DEBUG_LOG] Moons MLP: firstLoss=$firstLoss, lastLoss=$lastLoss")
        assertTrue(lastLoss < firstLoss * 0.5f, "Loss should drop by >=50% (was $firstLoss → $lastLoss)")

        // ------------------------------------------------------------------
        // 5. Eval: classification accuracy on held-out split via forward pass on a fresh inference ctx
        // ------------------------------------------------------------------
        val evalCtx = DirectCpuExecutionContext()
        val evalCount = evalY.size
        val xEval = evalCtx.fromFloatArray<FP32, Float>(Shape(evalCount, 2), FP32::class, evalX)
        val preds = model.forward(xEval, evalCtx)

        var correct = 0
        for (i in 0 until evalCount) {
            val score = preds.data.get(i, 0) as Float
            val predLabel = if (score >= 0f) 1f else -1f
            if (predLabel == evalY[i]) correct++
        }
        val accuracy = correct.toFloat() / evalCount
        println("[DEBUG_LOG] Held-out accuracy: $accuracy ($correct/$evalCount)")
        // Threshold deliberately set below the observed 17/20 = 0.85 to absorb small FP drift
        // in kotlin.math.tanh / .exp across JVMs while still asserting the model has learned
        // the moons decision boundary (random chance is 0.5).
        assertTrue(accuracy >= 0.80f, "Accuracy on moons should be >=0.80, got $accuracy")
    }

    // ----- Helpers: deterministic moons generator + z-score preprocessing -----

    /**
     * Mirror of `sklearn.datasets.make_moons` (shuffled). Returns:
     *   - Flat features `[x0_s0, x1_s0, x0_s1, x1_s1, ...]` of length `2 * nSamples`
     *   - Labels in {-1, +1} of length `nSamples`
     */
    private fun makeMoons(nSamples: Int, noise: Float, seed: Int): Pair<FloatArray, FloatArray> {
        val rng = Random(seed)
        val nOuter = nSamples / 2
        val nInner = nSamples - nOuter

        val features = FloatArray(nSamples * 2)
        val labels = FloatArray(nSamples)

        // Outer moon (upper half-circle), label +1
        for (i in 0 until nOuter) {
            val t = (i.toFloat() / nOuter) * PI.toFloat()
            features[2 * i] = cos(t) + gaussian(rng) * noise
            features[2 * i + 1] = sin(t) + gaussian(rng) * noise
            labels[i] = +1f
        }
        // Inner moon (shifted lower half-circle), label -1
        for (i in 0 until nInner) {
            val t = (i.toFloat() / nInner) * PI.toFloat()
            features[2 * (nOuter + i)] = 1f - cos(t) + gaussian(rng) * noise
            features[2 * (nOuter + i) + 1] = 0.5f - sin(t) + gaussian(rng) * noise
            labels[nOuter + i] = -1f
        }

        // Fisher-Yates shuffle by paired sample index (keep features and labels aligned)
        for (i in nSamples - 1 downTo 1) {
            val j = rng.nextInt(i + 1)
            // swap labels
            val tmpL = labels[i]; labels[i] = labels[j]; labels[j] = tmpL
            // swap feature pair
            val a0 = features[2 * i]; val a1 = features[2 * i + 1]
            features[2 * i] = features[2 * j]; features[2 * i + 1] = features[2 * j + 1]
            features[2 * j] = a0; features[2 * j + 1] = a1
        }
        return features to labels
    }

    /** Box-Muller standard normal — one sample per call. */
    private fun gaussian(rng: Random): Float {
        val u1 = rng.nextFloat().coerceAtLeast(1e-7f)
        val u2 = rng.nextFloat()
        return (sqrt(-2.0 * kotlin.math.ln(u1.toDouble())) * cos(2.0 * PI * u2)).toFloat()
    }

    private fun featurewiseMeanStd(flat: FloatArray, features: Int): Pair<FloatArray, FloatArray> {
        val n = flat.size / features
        val mean = FloatArray(features)
        val std = FloatArray(features)
        for (i in 0 until n) for (f in 0 until features) mean[f] += flat[i * features + f]
        for (f in 0 until features) mean[f] /= n
        for (i in 0 until n) for (f in 0 until features) {
            val d = flat[i * features + f] - mean[f]
            std[f] += d * d
        }
        for (f in 0 until features) std[f] = sqrt((std[f] / n).toDouble()).toFloat()
        return mean to std
    }

    private fun zScoreInPlace(flat: FloatArray, mean: FloatArray, std: FloatArray, features: Int) {
        val n = flat.size / features
        for (i in 0 until n) for (f in 0 until features) {
            val s = if (std[f] == 0f) 1f else std[f]
            flat[i * features + f] = (flat[i * features + f] - mean[f]) / s
        }
    }
}
