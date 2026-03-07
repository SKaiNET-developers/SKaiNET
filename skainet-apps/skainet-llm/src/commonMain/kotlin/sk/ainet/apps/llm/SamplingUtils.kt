package sk.ainet.apps.llm

import kotlin.math.exp
import kotlin.random.Random

/**
 * Numerically stable in-place softmax over the first [length] elements of [values].
 *
 * After this call, `values[0..length-1]` will sum to 1.0 (approximately).
 * Uses the max-subtraction trick to avoid overflow.
 */
public fun softmaxInPlace(values: FloatArray, length: Int = values.size) {
    var maxVal = Float.NEGATIVE_INFINITY
    for (i in 0 until length) {
        if (values[i] > maxVal) maxVal = values[i]
    }
    if (maxVal.isInfinite()) maxVal = 0f
    var sum = 0f
    for (i in 0 until length) {
        val e = exp((values[i] - maxVal).toDouble()).toFloat()
        values[i] = e
        sum += e
    }
    if (sum == 0f) return
    val inv = 1f / sum
    for (i in 0 until length) {
        values[i] *= inv
    }
}

/**
 * Sample a token ID from a logits array.
 *
 * **Warning:** This function mutates [logits] in-place (temperature scaling + softmax).
 * Pass a copy if you need to preserve the original values.
 *
 * - temperature <= 1e-6 -> greedy (argmax)
 * - otherwise -> temperature-scaled softmax + categorical sample
 *
 * @param logits Raw logits array (will be mutated for temperature > 0)
 * @param temperature Sampling temperature
 * @param random Random generator for sampling
 * @return The sampled index
 */
public fun sampleFromLogits(
    logits: FloatArray,
    temperature: Float,
    random: Random = Random.Default
): Int {
    // Greedy (argmax) for near-zero temperature
    if (temperature <= 1e-6f) {
        var best = 0
        var bestVal = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestVal) {
                bestVal = logits[i]
                best = i
            }
        }
        return best
    }

    // Temperature scaling
    var maxLogit = Float.NEGATIVE_INFINITY
    for (i in logits.indices) {
        val v = logits[i] / temperature
        logits[i] = v
        if (v > maxLogit) maxLogit = v
    }

    // Softmax + categorical sample (combined to skip normalization)
    var sum = 0f
    for (i in logits.indices) {
        val e = exp((logits[i] - maxLogit).toDouble()).toFloat()
        logits[i] = e
        sum += e
    }

    val r = random.nextFloat() * sum
    var acc = 0f
    for (i in logits.indices) {
        acc += logits[i]
        if (acc >= r) return i
    }
    return logits.lastIndex
}
