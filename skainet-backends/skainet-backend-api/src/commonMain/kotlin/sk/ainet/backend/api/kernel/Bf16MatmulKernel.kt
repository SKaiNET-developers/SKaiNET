package sk.ainet.backend.api.kernel

import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * FP32 input × BF16-packed weight matrix multiplication:
 *
 *   C(m, n) = A(m, k) · B(k, n)
 *
 * `A` and `C` are FP32 (`FloatArray`); `B` carries BFloat16 values
 * packed little-endian as 2 bytes per element in a `ByteArray`. This
 * matches the on-disk layout SafeTensors emits for BF16 tensors, so
 * callers can pass raw file bytes without an intermediate copy.
 *
 * ## Why a dedicated BF16 kernel
 *
 * BF16 has the same exponent range as FP32 (8 bits) and 7 mantissa
 * bits. Conversion to FP32 is bit-shift only:
 *
 *   `float_bits = (bf16_bits & 0xFFFF) << 16`
 *
 * Compared to the status-quo "dequant the whole tensor at load and run
 * FP32 matmul":
 *  - Halves the memory bandwidth of the B operand (2 B/elem vs 4 B/elem),
 *    which is the dominant cost on the largest matmuls (token
 *    embeddings, attention projections, MLP).
 *  - On ARMv8.6-A+ the `BFMMLA` instruction folds 4 BF16-mul plus FP32
 *    accumulates into one cycle — roughly 2× SGEMM throughput.
 *  - Drops the per-load dequant pass entirely.
 *
 * Outside the inner loop the semantics are identical to
 * [Fp32MatmulKernel] — the contract on shapes, offsets, the
 * caller-controlled zero-then-accumulate semantics, the alias rules, and
 * the "fully overwrite the `m × n` block of `out`" obligation all carry
 * over.
 *
 * ## Stride convention
 *
 * `A` and `out` use **float** strides (`aStride`, `outStride` count
 * FP32 elements) — matches [Fp32MatmulKernel].
 *
 * `B` uses **byte** offsets and strides (`bByteOffset`, `bByteStride`
 * count raw bytes) — matches the byte-packed convention from
 * [Q4KMatmulKernel] and avoids the foot-gun of mixing element-stride
 * and byte-stride accessors on the same `ByteArray`. For a contiguous
 * `(k, n)` BF16 matrix `bByteStride == n * 2`.
 */
public interface Bf16MatmulKernel : NarrowFloatMatmulKernel {

    override val codec: NarrowFloatCodec get() = Bf16Codec

    /**
     * @param a left operand `(m, k)`, row-major FP32, stride [aStride]
     *   floats per row.
     * @param aOffset element offset into [a] where the (0, 0) entry lives.
     * @param aStride distance in **floats** between consecutive rows of [a].
     * @param b right operand `(k, n)`, row-major, packed BF16
     *   little-endian (2 bytes per element). [bByteOffset] is the byte
     *   offset of the (0, 0) entry; [bByteStride] is the byte distance
     *   between consecutive rows of [b]. For a contiguous matrix
     *   `bByteStride == n * 2`.
     * @param bByteOffset byte offset into [b].
     * @param bByteStride byte distance between consecutive rows of [b].
     * @param out output `(m, n)`, row-major FP32, stride [outStride] floats per row.
     * @param outOffset element offset into [out].
     * @param outStride distance in **floats** between consecutive rows of [out].
     * @param m number of rows of A and C.
     * @param n number of columns of B and C.
     * @param k contraction dimension (cols of A == rows of B). `k == 0`
     *   zeros the `m × n` output block.
     */
    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int
    )
}
