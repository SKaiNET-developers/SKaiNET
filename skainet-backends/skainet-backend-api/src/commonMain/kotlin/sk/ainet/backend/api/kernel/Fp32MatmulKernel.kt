package sk.ainet.backend.api.kernel

/**
 * FP32 matrix multiplication kernel: `C(m, n) = A(m, k) · B(k, n)` in
 * row-major layout.
 *
 * This is a thin SPI between high-level tensor ops and the actual
 * numeric kernel that does the FLOPs. It exists so a SIMD-accelerated
 * `matmul` can be plugged in without re-implementing the rest of an
 * op-level backend, and so a hand-written kernel can be tested against
 * a scalar reference.
 *
 * Strides are in **floats** (not bytes) and let callers pass sub-blocks
 * of larger arrays without copying. For a contiguous matrix of shape
 * `(m, k)`, `aStride == k`. For a sub-block, `aStride` is the leading
 * dimension of the *parent* matrix.
 *
 * Implementations must NOT mutate `a` or `b`. They MAY assume the
 * arrays do not alias each other or `out`. Implementations MUST fully
 * overwrite the `m × n` block of `out` they're responsible for —
 * accumulator semantics are caller-controlled (e.g. zero `out` first if
 * you want C = A·B; pre-fill `out` if you want C += A·B and the kernel
 * is fused for that — no fused-accumulate kernel is in scope yet).
 */
public interface Fp32MatmulKernel {
    /**
     * @param a left operand `(m, k)`, row-major, with stride `aStride` along
     *   the leading (row) dimension.
     * @param aOffset element offset into [a] where the (0, 0) entry lives.
     * @param aStride distance in floats between consecutive rows of [a].
     *   For a contiguous matrix this equals `k`.
     * @param b right operand `(k, n)`, row-major, with stride `bStride`.
     * @param bOffset element offset into [b].
     * @param bStride distance in floats between consecutive rows of [b].
     *   For a contiguous matrix this equals `n`.
     * @param out output `(m, n)`, row-major, with stride `outStride`.
     * @param outOffset element offset into [out].
     * @param outStride distance in floats between consecutive rows of [out].
     *   For a contiguous matrix this equals `n`.
     * @param m number of rows of A and C.
     * @param n number of columns of B and C.
     * @param k contraction dimension (cols of A == rows of B).
     */
    public fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int
    )
}
