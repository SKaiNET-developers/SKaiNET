package sk.ainet.exec.kernel

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ScalarMatmulKernelTest {

    private fun reference(a: FloatArray, b: FloatArray, m: Int, n: Int, k: Int): FloatArray {
        val out = FloatArray(m * n)
        for (i in 0 until m) for (j in 0 until n) {
            var s = 0f
            for (kk in 0 until k) s += a[i * k + kk] * b[kk * n + j]
            out[i * n + j] = s
        }
        return out
    }

    private fun assertNearlyEquals(expected: FloatArray, actual: FloatArray, tol: Float = 1e-5f) {
        assertEquals(expected.size, actual.size, "length mismatch")
        for (i in expected.indices) {
            val diff = kotlin.math.abs(expected[i] - actual[i])
            assertEquals(true, diff < tol, "mismatch at $i: expected ${expected[i]} actual ${actual[i]} diff $diff")
        }
    }

    @Test
    fun small_2x3x4_contiguous() {
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) // shape [2, 4]
        val b = FloatArray(4 * 3) { it.toFloat() } // shape [4, 3]
        val out = FloatArray(2 * 3)
        ScalarMatmulKernel.matmul(
            a, 0, 4,
            b, 0, 3,
            out, 0, 3,
            m = 2, n = 3, k = 4
        )
        assertNearlyEquals(reference(a, b, 2, 3, 4), out)
    }

    @Test
    fun deterministic_random_8x16x32() {
        val rng = kotlin.random.Random(42)
        val a = FloatArray(8 * 32) { rng.nextFloat() - 0.5f }
        val b = FloatArray(32 * 16) { rng.nextFloat() - 0.5f }
        val out = FloatArray(8 * 16)
        ScalarMatmulKernel.matmul(
            a, 0, 32,
            b, 0, 16,
            out, 0, 16,
            m = 8, n = 16, k = 32
        )
        assertNearlyEquals(reference(a, b, 8, 16, 32), out)
    }

    @Test
    fun stride_supports_sub_blocks() {
        // Parent A is shape [4, 8]. Take rows 1..2 as a 2×8 sub-block.
        // aOffset = 1 * 8 = 8, aStride = 8 (parent leading dim).
        val parentA = FloatArray(4 * 8) { it.toFloat() }
        val b = FloatArray(8 * 3) { (it + 1).toFloat() }
        val out = FloatArray(2 * 3)
        ScalarMatmulKernel.matmul(
            parentA, 8, 8,
            b, 0, 3,
            out, 0, 3,
            m = 2, n = 3, k = 8
        )
        // Compare: extract rows 1..2 of parentA into a contiguous [2, 8] buffer,
        // run the reference.
        val subA = FloatArray(2 * 8) { idx -> parentA[8 + idx] }
        assertNearlyEquals(reference(subA, b, 2, 3, 8), out)
    }

    @Test
    fun out_stride_supports_partial_writes() {
        // Output is a sub-block of a larger 4×6 buffer; write a 2×3 result
        // at rows 1..2, cols 1..3.
        val a = FloatArray(2 * 5) { (it + 1).toFloat() }
        val b = FloatArray(5 * 3) { (it + 1).toFloat() }
        val parentOut = FloatArray(4 * 6)
        ScalarMatmulKernel.matmul(
            a, 0, 5,
            b, 0, 3,
            parentOut, /* row 1, col 1 */ 1 * 6 + 1, 6,
            m = 2, n = 3, k = 5
        )
        val expected = reference(a, b, 2, 3, 5)
        for (i in 0 until 2) for (j in 0 until 3) {
            assertEquals(expected[i * 3 + j], parentOut[(1 + i) * 6 + (1 + j)],
                "output (sub-block) mismatch at parent[${1+i}][${1+j}]")
        }
        // Outside the written sub-block, parentOut is still zero.
        for (i in 0 until 4) for (j in 0 until 6) {
            val inside = i in 1..2 && j in 1..3
            if (!inside) {
                assertEquals(0f, parentOut[i * 6 + j], "parent[$i][$j] should be untouched")
            }
        }
    }

    @Test
    fun zero_m_or_n_no_op() {
        val a = FloatArray(0)
        val b = FloatArray(0)
        val out = FloatArray(5) { 7f }
        ScalarMatmulKernel.matmul(a, 0, 0, b, 0, 0, out, 0, 0, m = 0, n = 5, k = 0)
        for (v in out) assertEquals(7f, v, "out should be unchanged when m == 0")
    }

    @Test
    fun zero_k_zeros_output() {
        val out = FloatArray(2 * 3) { 9f }
        ScalarMatmulKernel.matmul(
            FloatArray(0), 0, 0,
            FloatArray(0), 0, 0,
            out, 0, 3,
            m = 2, n = 3, k = 0
        )
        for (v in out) assertEquals(0f, v, "out block should be zeroed when k == 0")
    }

    @Test
    fun rejects_negative_dimensions() {
        val a = FloatArray(0); val b = FloatArray(0); val out = FloatArray(0)
        assertFailsWith<IllegalArgumentException> {
            ScalarMatmulKernel.matmul(a, 0, 0, b, 0, 0, out, 0, 0, m = -1, n = 1, k = 1)
        }
    }
}
