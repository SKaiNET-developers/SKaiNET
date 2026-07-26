package sk.ainet.backend.api.kernel

import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * Matmul over a **16-bit float** right operand: `C(m,n) = A(m,k) · B(k,n)` where `B` is packed at
 * 2 bytes per element and `A`/`C` are FP32.
 *
 * The shape, offset, aliasing and "fully overwrite the `m × n` block" obligations are identical to
 * `Fp32MatmulKernel`. Only the inner-loop decode of `B` differs, which is what [codec] selects — so
 * BF16 and FP16 differ by a decode step and nothing else.
 *
 * **Accumulation is always FP32.** The narrow format is a storage width, never an accumulate width;
 * every implementation widens each `B` element before the multiply and sums in `Float`. This
 * matches what PyTorch, JAX and tensor cores do, and is not a shortcut to be "optimized" away.
 *
 * ## Stride convention
 *
 * `A` and `out` use **float** strides (counting FP32 elements). `B` uses **byte** offsets and
 * strides, matching the byte-packed convention of the block-quantized kernels and avoiding the
 * foot-gun of mixing element- and byte-strides on the same `ByteArray`. For a contiguous `(k, n)`
 * matrix, `bByteStride == n * 2`.
 */
public interface NarrowFloatMatmulKernel {

    /** The 16-bit format this kernel decodes `B` from. */
    public val codec: NarrowFloatCodec

    /**
     * @param a left operand `(m, k)`, row-major FP32, [aStride] floats per row.
     * @param aOffset element offset into [a] of the (0, 0) entry.
     * @param aStride distance in **floats** between consecutive rows of [a].
     * @param b right operand `(k, n)`, row-major, packed little-endian 16-bit floats.
     * @param bByteOffset **byte** offset into [b] of the (0, 0) entry.
     * @param bByteStride **byte** distance between consecutive rows of [b].
     * @param out output `(m, n)`, row-major FP32, [outStride] floats per row.
     * @param outOffset element offset into [out].
     * @param outStride distance in **floats** between consecutive rows of [out].
     * @param m rows of A and C.
     * @param n columns of B and C.
     * @param k contraction dimension. `k == 0` zeros the `m × n` output block.
     */
    public fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    )
}

/**
 * Matmul over an **IEEE binary16** right operand.
 *
 * The FP16 counterpart to [Bf16MatmulKernel]. Note that FP16 cannot use the bit-shift decode that
 * makes BF16 nearly free — binary16 needs exponent rebiasing — so vectorized FP16 implementations
 * lag their BF16 equivalents. Correctness first; throughput is a separate concern.
 */
public interface Fp16MatmulKernel : NarrowFloatMatmulKernel {
    override val codec: NarrowFloatCodec get() = Fp16Codec
}
