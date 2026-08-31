package sk.ainet.exec.kernel

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import sk.ainet.backend.api.kernel.Fp32MatmulKernel

/**
 * SIMD reference [Fp32MatmulKernel] implemented on the JDK Vector API
 * (JEP 338+, `jdk.incubator.vector`). Produces results that match
 * [ScalarMatmulKernel] within FP-rounding tolerance.
 *
 * Strategy:
 * - Pack `B` into a transposed buffer `bt` of shape `(n, k)` so the
 *   inner reduction streams contiguously over `k` for both operands.
 * - Cache-block the `(m, n, k)` iteration space with tiles
 *   ([TILE_M], [TILE_N], [TILE_K]). Default 8×8×128 keeps a working
 *   set well under L1 — eight A rows × 128 floats + eight Bᵀ rows ×
 *   128 floats ≈ 8 KB, within typical 32 KB L1.
 * - Within each (TILE_M × TILE_N) sub-tile, [mnpack] recursively
 *   dispatches into `RM × RN` micro-kernels — `gemm4x3`, `gemm2x2`,
 *   `gemm2x1`, `gemm1x2`, `gemm1x1`. Each micro-kernel keeps
 *   `RM × RN` `FloatVector` accumulators in locals and amortizes
 *   every A-row load across `RN` columns and every B-column load
 *   across `RM` rows. This mirrors the tile-dispatch pattern from
 *   tinyBLAS (`sgemm.cpp`, Justine Tunney / llamafile).
 * - On AVX2 the largest microkernel that fits inside 16 YMM registers
 *   is `4 × 3` (12 accumulators + at most 4 A vectors + 1 B vector
 *   live at once). Smaller microkernels cover residual rows and
 *   columns that don't divide evenly into the larger tile shape.
 * - Inner reduction is a vector-width FMA accumulator
 *   (`v.fma(w, acc)`), reduced via `reduceLanes(ADD)` once per
 *   `(i, j)` cell per K-tile. Tail elements that don't fill a vector
 *   lane are handled in scalar.
 * - Output is zeroed once up front; per-tile work accumulates via `+=`
 *   so the K-loop can split across multiple tiles cleanly.
 *
 * The B-pack is `O(n * k)` floats per call; cheap relative to the
 * `O(m * n * k)` FLOPs but still allocates each invocation. A
 * scratch-pool integration is out of scope for this kernel and lives
 * one layer up (see `ScratchPool` SPI in `skainet-lang-core`).
 *
 * ## Few rows take a different path
 *
 * Both fixed costs above are `O(n * k)` and independent of `m`: packing Bᵀ, and a horizontal
 * `reduceLanes` per output cell per K-tile. They buy nothing when there are few rows to amortize
 * them over, and a decode step is exactly that — `m = 1`. Measured at `k=1536, n=256` on an Apple
 * M4 before [gemvRows] existed, this kernel was **slower than [ScalarMatmulKernel]**, 0.969 ms
 * against 0.253. So at or below [GEMV_MAX_M] rows it accumulates by outer product instead, which
 * touches B once, contiguously, and never reduces across lanes:
 *
 * ```
 *   m     tiled (before)   gemvRows (after)     native-ffm    scalar
 *    1    0.969 ms  0.81    0.058 ms  13.63     7.65          3.05     GFLOP/s
 *    4       —               0.230 ms  13.70    18.50         3.07
 *    8       —               0.463 ms  13.58    21.41         3.02
 *   16    1.332 ms  9.45    (tiled)              25.53         3.06
 *   32    1.647 ms 15.28    (tiled)              27.69         3.10
 * ```
 *
 * The tiled path stays in charge above that, where it is what the shape wants. Note the native FFM
 * kernel wins from `m = 4` up but loses at `m = 1`, where the heap→off-heap copy a JDK 21 downcall
 * requires costs more than the arithmetic; `Fp32KernelRaceBench` keeps these numbers honest.
 *
 * Caller contract is identical to [Fp32MatmulKernel]: strides are in
 * floats, `out` is fully overwritten in the `m × n` block, and `k == 0`
 * zeros the output block.
 */
public object PanamaVectorMatmulKernel : Fp32MatmulKernel {
    private val species: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    private const val TILE_M = 8
    private const val TILE_N = 8
    private const val TILE_K = 128

    /**
     * At or below this many rows the tiled path's O(n*k) setup outweighs its O(m*n*k) speedup, so
     * [gemvRows] serves instead. Chosen from measurement, not theory — see `Fp32KernelRaceBench`.
     */
    private const val GEMV_MAX_M = 8

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "PanamaVectorMatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return
        // Zero the m×n output block once. The K-tile loop accumulates
        // via `+=`, so the contract "fully overwrite the output block"
        // is preserved even when k == 0 (no tile loop runs).
        for (i in 0 until m) {
            val rowOff = outOffset + i * outStride
            for (j in 0 until n) out[rowOff + j] = 0f
        }
        if (k == 0) return

        // Few rows: skip the tiled path entirely. Both of its fixed costs are O(n*k) — packing B
        // transposed, and a horizontal reduceLanes per output cell per K-tile — so at small m they
        // dwarf the O(m*n*k) arithmetic they exist to accelerate. A decode step is m=1, and there
        // this kernel measured 0.969 ms against 0.253 for the scalar one it is supposed to beat.
        if (m <= GEMV_MAX_M) {
            gemvRows(a, aOffset, aStride, b, bOffset, bStride, out, outOffset, outStride, m, n, k)
            return
        }

        // Pack B^T: bt[j, kk] = b[kk, j]. Row stride in bt is k.
        val bt = FloatArray(n * k)
        for (kk in 0 until k) {
            val src = bOffset + kk * bStride
            for (j in 0 until n) {
                bt[j * k + kk] = b[src + j]
            }
        }

        var mTile = 0
        while (mTile < m) {
            val mEnd = minOf(mTile + TILE_M, m)
            var nTile = 0
            while (nTile < n) {
                val nEnd = minOf(nTile + TILE_N, n)
                var kTile = 0
                while (kTile < k) {
                    val kEnd = minOf(kTile + TILE_K, k)
                    mnpack(
                        a, aOffset, aStride,
                        bt, k,
                        out, outOffset, outStride,
                        mTile, mEnd, nTile, nEnd,
                        kTile, kEnd - kTile,
                    )
                    kTile = kEnd
                }
                nTile = nEnd
            }
            mTile = mEnd
        }
    }

    /**
     * Outer-product accumulation over `out` rows: `out[i, :] += a[i, p] * b[p, :]`.
     *
     * Streams `b` and `out` contiguously along `n` and never transposes or packs anything, so the
     * whole call costs one pass over B. There is no horizontal reduction — each lane owns one
     * output column for the entire `k` loop — which is the other thing the tiled path pays per
     * cell. Accumulation for a given output stays in ascending `p`, matching the scalar kernel's
     * order rather than the tiled path's split-by-K-tile order.
     */
    private fun gemvRows(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        val step = species.length()
        val bound = species.loopBound(n)
        for (i in 0 until m) {
            val aBase = aOffset + i * aStride
            val outRow = outOffset + i * outStride
            for (p in 0 until k) {
                val av = a[aBase + p]
                if (av == 0f) continue
                val vav = FloatVector.broadcast(species, av)
                val bRow = bOffset + p * bStride
                var j = 0
                while (j < bound) {
                    val vo = FloatVector.fromArray(species, out, outRow + j)
                    vav.fma(FloatVector.fromArray(species, b, bRow + j), vo).intoArray(out, outRow + j)
                    j += step
                }
                while (j < n) {
                    out[outRow + j] += av * b[bRow + j]
                    j++
                }
            }
        }
    }

    /**
     * Recursive (m, n) tile dispatch. Picks the largest microkernel
     * shape `(RM, RN)` that fits the residual `(m1-m0, n1-n0)`, calls it
     * over the aligned sub-rectangle `[m0..mp) × [n0..np)`, then recurses
     * on the residual rows `[mp..m1) × [n0..np)` and the residual columns
     * `[m0..m1) × [np..n1)`. Mirrors the tinyBLAS `mnpack` switch but
     * uses only the AVX2-friendly microkernel set (16 vector registers).
     */
    private fun mnpack(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        if (m1 <= m0 || n1 <= n0) return

        val rm = minOf(m1 - m0, 4)
        val rn = minOf(n1 - n0, 3)
        val mc: Int
        val nc: Int
        when ((rm shl 4) or rn) {
            0x43 -> {
                mc = 4; nc = 3
                gemm4x3(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
                    m0, m0 + ((m1 - m0) / mc) * mc, n0, n0 + ((n1 - n0) / nc) * nc, kStart, kLen)
            }
            0x42, 0x33, 0x32, 0x23, 0x22 -> {
                mc = 2; nc = 2
                gemm2x2(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
                    m0, m0 + ((m1 - m0) / mc) * mc, n0, n0 + ((n1 - n0) / nc) * nc, kStart, kLen)
            }
            0x41, 0x31, 0x21 -> {
                mc = 2; nc = 1
                gemm2x1(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
                    m0, m0 + ((m1 - m0) / mc) * mc, n0, n0 + ((n1 - n0) / nc) * nc, kStart, kLen)
            }
            0x13, 0x12 -> {
                mc = 1; nc = 2
                gemm1x2(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
                    m0, m0 + ((m1 - m0) / mc) * mc, n0, n0 + ((n1 - n0) / nc) * nc, kStart, kLen)
            }
            0x11 -> {
                mc = 1; nc = 1
                gemm1x1(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
                    m0, m0 + ((m1 - m0) / mc) * mc, n0, n0 + ((n1 - n0) / nc) * nc, kStart, kLen)
            }
            else -> return
        }
        val mp = m0 + ((m1 - m0) / mc) * mc
        val np = n0 + ((n1 - n0) / nc) * nc
        if (mp < m1) mnpack(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
            mp, m1, n0, np, kStart, kLen)
        if (np < n1) mnpack(a, aOffset, aStride, bt, btStride, out, outOffset, outStride,
            m0, m1, np, n1, kStart, kLen)
    }

    /**
     * Largest AVX2-friendly microkernel: 4 rows × 3 cols, 12 accumulators.
     * Loads 4 A vectors and 3 B vectors per `k` step, issues 12 FMAs.
     * Caller guarantees `(m1 - m0)` is a multiple of 4 and `(n1 - n0)` of 3.
     */
    private fun gemm4x3(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        val step = species.length()
        val loopBound = species.loopBound(kLen)
        var ii = m0
        while (ii < m1) {
            val a0Base = aOffset + ii * aStride + kStart
            val a1Base = a0Base + aStride
            val a2Base = a1Base + aStride
            val a3Base = a2Base + aStride
            val outRow0 = outOffset + ii * outStride
            val outRow1 = outRow0 + outStride
            val outRow2 = outRow1 + outStride
            val outRow3 = outRow2 + outStride
            var jj = n0
            while (jj < n1) {
                val b0Base = jj * btStride + kStart
                val b1Base = b0Base + btStride
                val b2Base = b1Base + btStride

                var c00 = FloatVector.zero(species); var c01 = FloatVector.zero(species); var c02 = FloatVector.zero(species)
                var c10 = FloatVector.zero(species); var c11 = FloatVector.zero(species); var c12 = FloatVector.zero(species)
                var c20 = FloatVector.zero(species); var c21 = FloatVector.zero(species); var c22 = FloatVector.zero(species)
                var c30 = FloatVector.zero(species); var c31 = FloatVector.zero(species); var c32 = FloatVector.zero(species)

                var idx = 0
                while (idx < loopBound) {
                    val va0 = FloatVector.fromArray(species, a, a0Base + idx)
                    val va1 = FloatVector.fromArray(species, a, a1Base + idx)
                    val va2 = FloatVector.fromArray(species, a, a2Base + idx)
                    val va3 = FloatVector.fromArray(species, a, a3Base + idx)

                    val vb0 = FloatVector.fromArray(species, bt, b0Base + idx)
                    c00 = va0.fma(vb0, c00); c10 = va1.fma(vb0, c10); c20 = va2.fma(vb0, c20); c30 = va3.fma(vb0, c30)

                    val vb1 = FloatVector.fromArray(species, bt, b1Base + idx)
                    c01 = va0.fma(vb1, c01); c11 = va1.fma(vb1, c11); c21 = va2.fma(vb1, c21); c31 = va3.fma(vb1, c31)

                    val vb2 = FloatVector.fromArray(species, bt, b2Base + idx)
                    c02 = va0.fma(vb2, c02); c12 = va1.fma(vb2, c12); c22 = va2.fma(vb2, c22); c32 = va3.fma(vb2, c32)

                    idx += step
                }

                var s00 = c00.reduceLanes(VectorOperators.ADD); var s01 = c01.reduceLanes(VectorOperators.ADD); var s02 = c02.reduceLanes(VectorOperators.ADD)
                var s10 = c10.reduceLanes(VectorOperators.ADD); var s11 = c11.reduceLanes(VectorOperators.ADD); var s12 = c12.reduceLanes(VectorOperators.ADD)
                var s20 = c20.reduceLanes(VectorOperators.ADD); var s21 = c21.reduceLanes(VectorOperators.ADD); var s22 = c22.reduceLanes(VectorOperators.ADD)
                var s30 = c30.reduceLanes(VectorOperators.ADD); var s31 = c31.reduceLanes(VectorOperators.ADD); var s32 = c32.reduceLanes(VectorOperators.ADD)

                while (idx < kLen) {
                    val av0 = a[a0Base + idx]; val av1 = a[a1Base + idx]; val av2 = a[a2Base + idx]; val av3 = a[a3Base + idx]
                    val bv0 = bt[b0Base + idx]; val bv1 = bt[b1Base + idx]; val bv2 = bt[b2Base + idx]
                    s00 += av0 * bv0; s10 += av1 * bv0; s20 += av2 * bv0; s30 += av3 * bv0
                    s01 += av0 * bv1; s11 += av1 * bv1; s21 += av2 * bv1; s31 += av3 * bv1
                    s02 += av0 * bv2; s12 += av1 * bv2; s22 += av2 * bv2; s32 += av3 * bv2
                    idx++
                }

                out[outRow0 + jj] += s00; out[outRow0 + jj + 1] += s01; out[outRow0 + jj + 2] += s02
                out[outRow1 + jj] += s10; out[outRow1 + jj + 1] += s11; out[outRow1 + jj + 2] += s12
                out[outRow2 + jj] += s20; out[outRow2 + jj + 1] += s21; out[outRow2 + jj + 2] += s22
                out[outRow3 + jj] += s30; out[outRow3 + jj + 1] += s31; out[outRow3 + jj + 2] += s32

                jj += 3
            }
            ii += 4
        }
    }

    /** 2 × 2 microkernel: 4 accumulators, 2 A loads + 2 B loads + 4 FMAs per step. */
    private fun gemm2x2(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        val step = species.length()
        val loopBound = species.loopBound(kLen)
        var ii = m0
        while (ii < m1) {
            val a0Base = aOffset + ii * aStride + kStart
            val a1Base = a0Base + aStride
            val outRow0 = outOffset + ii * outStride
            val outRow1 = outRow0 + outStride
            var jj = n0
            while (jj < n1) {
                val b0Base = jj * btStride + kStart
                val b1Base = b0Base + btStride

                var c00 = FloatVector.zero(species); var c01 = FloatVector.zero(species)
                var c10 = FloatVector.zero(species); var c11 = FloatVector.zero(species)

                var idx = 0
                while (idx < loopBound) {
                    val va0 = FloatVector.fromArray(species, a, a0Base + idx)
                    val va1 = FloatVector.fromArray(species, a, a1Base + idx)
                    val vb0 = FloatVector.fromArray(species, bt, b0Base + idx)
                    val vb1 = FloatVector.fromArray(species, bt, b1Base + idx)
                    c00 = va0.fma(vb0, c00); c10 = va1.fma(vb0, c10)
                    c01 = va0.fma(vb1, c01); c11 = va1.fma(vb1, c11)
                    idx += step
                }

                var s00 = c00.reduceLanes(VectorOperators.ADD); var s01 = c01.reduceLanes(VectorOperators.ADD)
                var s10 = c10.reduceLanes(VectorOperators.ADD); var s11 = c11.reduceLanes(VectorOperators.ADD)

                while (idx < kLen) {
                    val av0 = a[a0Base + idx]; val av1 = a[a1Base + idx]
                    val bv0 = bt[b0Base + idx]; val bv1 = bt[b1Base + idx]
                    s00 += av0 * bv0; s10 += av1 * bv0
                    s01 += av0 * bv1; s11 += av1 * bv1
                    idx++
                }

                out[outRow0 + jj] += s00; out[outRow0 + jj + 1] += s01
                out[outRow1 + jj] += s10; out[outRow1 + jj + 1] += s11

                jj += 2
            }
            ii += 2
        }
    }

    /** 2 × 1 microkernel: 2 accumulators, 2 A loads + 1 B load + 2 FMAs per step. */
    private fun gemm2x1(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        val step = species.length()
        val loopBound = species.loopBound(kLen)
        var ii = m0
        while (ii < m1) {
            val a0Base = aOffset + ii * aStride + kStart
            val a1Base = a0Base + aStride
            val outRow0 = outOffset + ii * outStride
            val outRow1 = outRow0 + outStride
            for (jj in n0 until n1) {
                val b0Base = jj * btStride + kStart

                var c0 = FloatVector.zero(species)
                var c1 = FloatVector.zero(species)

                var idx = 0
                while (idx < loopBound) {
                    val va0 = FloatVector.fromArray(species, a, a0Base + idx)
                    val va1 = FloatVector.fromArray(species, a, a1Base + idx)
                    val vb = FloatVector.fromArray(species, bt, b0Base + idx)
                    c0 = va0.fma(vb, c0); c1 = va1.fma(vb, c1)
                    idx += step
                }

                var s0 = c0.reduceLanes(VectorOperators.ADD)
                var s1 = c1.reduceLanes(VectorOperators.ADD)

                while (idx < kLen) {
                    val bv = bt[b0Base + idx]
                    s0 += a[a0Base + idx] * bv
                    s1 += a[a1Base + idx] * bv
                    idx++
                }

                out[outRow0 + jj] += s0
                out[outRow1 + jj] += s1
            }
            ii += 2
        }
    }

    /** 1 × 2 microkernel: 2 accumulators, 1 A load + 2 B loads + 2 FMAs per step. */
    private fun gemm1x2(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        val step = species.length()
        val loopBound = species.loopBound(kLen)
        for (ii in m0 until m1) {
            val aBase = aOffset + ii * aStride + kStart
            val outRow = outOffset + ii * outStride
            var jj = n0
            while (jj < n1) {
                val b0Base = jj * btStride + kStart
                val b1Base = b0Base + btStride

                var c0 = FloatVector.zero(species)
                var c1 = FloatVector.zero(species)

                var idx = 0
                while (idx < loopBound) {
                    val va = FloatVector.fromArray(species, a, aBase + idx)
                    val vb0 = FloatVector.fromArray(species, bt, b0Base + idx)
                    val vb1 = FloatVector.fromArray(species, bt, b1Base + idx)
                    c0 = va.fma(vb0, c0); c1 = va.fma(vb1, c1)
                    idx += step
                }

                var s0 = c0.reduceLanes(VectorOperators.ADD)
                var s1 = c1.reduceLanes(VectorOperators.ADD)

                while (idx < kLen) {
                    val av = a[aBase + idx]
                    s0 += av * bt[b0Base + idx]
                    s1 += av * bt[b1Base + idx]
                    idx++
                }

                out[outRow + jj] += s0
                out[outRow + jj + 1] += s1

                jj += 2
            }
        }
    }

    /** 1 × 1 microkernel: single-cell fallback. Equivalent to the pre-change inner loop. */
    private fun gemm1x1(
        a: FloatArray, aOffset: Int, aStride: Int,
        bt: FloatArray, btStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m0: Int, m1: Int, n0: Int, n1: Int,
        kStart: Int, kLen: Int,
    ) {
        val step = species.length()
        val loopBound = species.loopBound(kLen)
        for (ii in m0 until m1) {
            val aBase = aOffset + ii * aStride + kStart
            val outRow = outOffset + ii * outStride
            for (jj in n0 until n1) {
                val bBase = jj * btStride + kStart
                var acc = FloatVector.zero(species)
                var idx = 0
                while (idx < loopBound) {
                    val va = FloatVector.fromArray(species, a, aBase + idx)
                    val vb = FloatVector.fromArray(species, bt, bBase + idx)
                    acc = va.fma(vb, acc)
                    idx += step
                }
                var sum = acc.reduceLanes(VectorOperators.ADD)
                while (idx < kLen) {
                    sum += a[aBase + idx] * bt[bBase + idx]
                    idx++
                }
                out[outRow + jj] += sum
            }
        }
    }
}
