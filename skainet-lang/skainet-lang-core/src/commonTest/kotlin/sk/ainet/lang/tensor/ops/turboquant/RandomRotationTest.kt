package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RandomRotationTest {

    @Test
    fun rotateInverseRoundTrip() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val original = input.copyOf()
        val seed = 42

        RandomRotation.rotate(input, seed)
        RandomRotation.inverseRotate(input, seed)

        for (i in original.indices) {
            assertTrue(abs(original[i] - input[i]) < 1e-4f,
                "Element $i: expected ${original[i]}, got ${input[i]}")
        }
    }

    @Test
    fun rotateChangesValues() {
        val input = floatArrayOf(1f, 0f, 0f, 0f)
        val original = input.copyOf()

        RandomRotation.rotate(input, 42)

        // At least some values should change
        var changed = false
        for (i in input.indices) {
            if (abs(input[i] - original[i]) > 1e-6f) changed = true
        }
        assertTrue(changed, "Rotation should modify the vector")
    }

    @Test
    fun rotateDeterministic() {
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val b = floatArrayOf(1f, 2f, 3f, 4f)

        RandomRotation.rotate(a, 123)
        RandomRotation.rotate(b, 123)

        assertTrue(a.contentEquals(b), "Same seed should produce same rotation")
    }

    @Test
    fun rotatePreservesNorm() {
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val normBefore = sqrt(input.sumOf { (it * it).toDouble() }).toFloat()

        RandomRotation.rotate(input, 42)

        val normAfter = sqrt(input.sumOf { (it * it).toDouble() }).toFloat()
        // WHT preserves norm (orthogonal transform)
        assertTrue(abs(normBefore - normAfter) < 0.1f * normBefore,
            "Norm should be approximately preserved: before=$normBefore, after=$normAfter")
    }

    @Test
    fun seedForIsDeterministic() {
        val s1 = RandomRotation.seedFor(0, 1, 2)
        val s2 = RandomRotation.seedFor(0, 1, 2)
        assertEquals(s1, s2)
    }

    @Test
    fun seedForDistribution() {
        // Different inputs should produce different seeds
        val seeds = mutableSetOf<Int>()
        for (l in 0..3) {
            for (h in 0..3) {
                for (p in 0..3) {
                    seeds.add(RandomRotation.seedFor(l, h, p))
                }
            }
        }
        // 64 inputs should produce at least 50 distinct seeds (well-distributed)
        assertTrue(seeds.size > 50, "Seeds should be well-distributed, got ${seeds.size} unique out of 64")
    }

    @Test
    fun walshHadamardSmall() {
        // WHT of [1, 1, 1, 1] should give [2, 0, 0, 0] (before normalization: [4, 0, 0, 0])
        // After normalization by 1/sqrt(4) = 0.5: [2, 0, 0, 0]
        val input = floatArrayOf(1f, 1f, 1f, 1f)
        RandomRotation.walshHadamard(input)
        assertTrue(abs(input[0] - 2f) < 1e-5f, "WHT[0] should be 2, got ${input[0]}")
        assertTrue(abs(input[1]) < 1e-5f, "WHT[1] should be 0, got ${input[1]}")
    }
}
