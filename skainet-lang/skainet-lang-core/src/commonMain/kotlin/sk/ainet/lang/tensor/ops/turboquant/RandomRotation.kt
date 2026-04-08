package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Random rotation for TurboQuant encoding.
 *
 * TurboQuant uses random orthogonal rotations to spread quantization error
 * uniformly across dimensions before scalar quantization. This is the first
 * stage of the TurboQuant pipeline.
 *
 * The rotation is **deterministic** given a seed, so the same rotation can
 * be reproduced during decoding without storing the full rotation matrix.
 *
 * Implementation uses fast random Hadamard-like rotations (random sign flips
 * + structured permutation) rather than full O(d^2) matrix multiplication.
 * This gives O(d log d) rotation cost.
 */
public object RandomRotation {

    /**
     * Apply a seeded random rotation to a vector in-place.
     *
     * Uses the "random sign flip + fast Walsh-Hadamard transform" approach:
     * 1. Apply random +-1 sign flips (seeded)
     * 2. Apply normalized Walsh-Hadamard transform
     *
     * This produces a near-uniform rotation in O(d log d) time.
     *
     * @param vector  Input/output vector (modified in place)
     * @param seed    Deterministic seed for reproducibility
     */
    public fun rotate(vector: FloatArray, seed: Int) {
        randomSignFlip(vector, seed)
        walshHadamard(vector)
    }

    /**
     * Apply the inverse rotation to recover the original vector.
     *
     * Since sign flips and Hadamard are both self-inverse (up to normalization),
     * the inverse is the same operations in reverse order.
     *
     * @param vector  Input/output vector (modified in place)
     * @param seed    Same seed used during [rotate]
     */
    public fun inverseRotate(vector: FloatArray, seed: Int) {
        walshHadamard(vector)
        randomSignFlip(vector, seed)
    }

    /**
     * Apply random +-1 sign flips to each element.
     *
     * This is equivalent to multiplying by a diagonal matrix D where
     * D_ii ∈ {-1, +1} drawn from a seeded PRNG.
     */
    internal fun randomSignFlip(vector: FloatArray, seed: Int) {
        val rng = Random(seed)
        for (i in vector.indices) {
            if (rng.nextBoolean()) {
                vector[i] = -vector[i]
            }
        }
    }

    /**
     * In-place normalized Walsh-Hadamard transform.
     *
     * The WHT is an orthogonal transform (when normalized by 1/sqrt(n))
     * that can be computed in O(n log n) time. It spreads information
     * uniformly across all dimensions.
     *
     * For non-power-of-2 dimensions, the vector is conceptually zero-padded
     * to the next power of 2, transformed, then truncated. In practice we
     * handle this by processing only up to the largest power of 2 <= n and
     * leaving remaining elements with just the sign flip.
     */
    internal fun walshHadamard(vector: FloatArray) {
        val n = vector.size
        if (n <= 1) return

        // Find largest power of 2 <= n
        var len = 1
        while (len * 2 <= n) len *= 2

        // Iterative WHT (butterfly)
        var h = 1
        while (h < len) {
            var i = 0
            while (i < len) {
                for (j in i until i + h) {
                    val x = vector[j]
                    val y = vector[j + h]
                    vector[j] = x + y
                    vector[j + h] = x - y
                }
                i += h * 2
            }
            h *= 2
        }

        // Normalize by 1/sqrt(len) to make the transform orthogonal
        val norm = 1.0f / sqrt(len.toFloat())
        for (i in 0 until len) {
            vector[i] *= norm
        }
    }

    /**
     * Generate a rotation seed for a given (layer, head, position) triple.
     *
     * Uses a simple hash combining function to produce deterministic seeds
     * that are well-distributed across the seed space.
     */
    public fun seedFor(layer: Int, head: Int, position: Int): Int {
        var h = layer
        h = h * 31 + head
        h = h * 31 + position
        // Mix bits (MurmurHash3 finalizer)
        h = h xor (h ushr 16)
        h *= 0x85ebca6b.toInt()
        h = h xor (h ushr 13)
        h *= 0xc2b2ae35.toInt()
        h = h xor (h ushr 16)
        return h
    }
}
