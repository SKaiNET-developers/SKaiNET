package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Fp16MatmulKernel

/**
 * Native (FFM) implementation of [Fp16MatmulKernel].
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_fp16_matmul(
 *       const float* a,   int32_t a_offset,      int32_t a_stride,
 *       const uint8_t* b, int32_t b_byte_offset, int32_t b_byte_stride,
 *       float* c,         int32_t c_offset,      int32_t c_stride,
 *       int32_t m, int32_t n, int32_t k);
 *
 * Mirrors [NativeBf16MatmulKernel] exactly apart from the symbol and the
 * dequant the C side performs: binary16 needs exponent rebiasing and
 * gradual-underflow handling where BF16 needs one shift, so the C kernel does
 * it branch-free to keep the inner loop vectorizable.
 *
 * **Why this exists (#887).** Until it did, [NativeKernelProvider] carried
 * `matmulBf16` but not `matmulFp16`, so BF16 resolved to a native kernel while
 * FP16 fell through to `PanamaVectorFp16MatmulKernel` at priority 50. That
 * asymmetry — not the cost of the decode, which is within ~15% between the two
 * Panama kernels — is what made FP16 measure 2-18x slower than the FP32 SGEMM
 * while BF16 measured faster than it.
 *
 * Numerical parity vs `ScalarFp16MatmulKernel` is asserted by
 * `NativeFp16MatmulKernelParityTest` within FMA + reordered-reduction
 * tolerance — the same bar the BF16 and FP32 native parity tests use.
 */
internal object NativeFp16MatmulKernel : Fp16MatmulKernel {

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "NativeFp16MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return

        val mh = handle
            ?: error("NativeFp16MatmulKernel.matmul invoked while native library unavailable")

        // Reach calculations. For non-contiguous strides we may skip past
        // unused elements; allocating to the full reach keeps the kernel's
        // pointer arithmetic simple and matches the Kotlin-side bounds.
        val aReachFloats = if (m == 0 || k == 0) 0 else aOffset + (m - 1) * aStride + k
        val bReachBytes = if (k == 0 || n == 0) 0
                          else bByteOffset + (k - 1) * bByteStride + n * 2
        val cReachFloats = outOffset + (m - 1) * outStride + n

        Arena.ofConfined().use { arena ->
            val aBytes = aReachFloats.toLong() * java.lang.Float.BYTES
            val bBytes = bReachBytes.toLong()
            val cBytes = cReachFloats.toLong() * java.lang.Float.BYTES
            val fAlign = ValueLayout.JAVA_FLOAT.byteAlignment()
            val bAlign = ValueLayout.JAVA_BYTE.byteAlignment()

            val aSeg: MemorySegment = if (aBytes > 0) arena.allocate(aBytes, fAlign) else MemorySegment.NULL
            val bSeg: MemorySegment = if (bBytes > 0) arena.allocate(bBytes, bAlign) else MemorySegment.NULL
            val cSeg: MemorySegment = arena.allocate(cBytes, fAlign)

            if (aReachFloats > 0) {
                MemorySegment.copy(a, 0, aSeg, ValueLayout.JAVA_FLOAT, 0L, aReachFloats)
            }
            if (bReachBytes > 0) {
                MemorySegment.copy(b, 0, bSeg, ValueLayout.JAVA_BYTE, 0L, bReachBytes)
            }

            mh.invoke(
                aSeg, aOffset, aStride,
                bSeg, bByteOffset, bByteStride,
                cSeg, outOffset, outStride,
                m, n, k,
            )

            MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, cReachFloats)
        }
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_fp16_matmul").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // a
            ValueLayout.JAVA_INT,   // a_offset
            ValueLayout.JAVA_INT,   // a_stride
            ValueLayout.ADDRESS,    // b
            ValueLayout.JAVA_INT,   // b_byte_offset
            ValueLayout.JAVA_INT,   // b_byte_stride
            ValueLayout.ADDRESS,    // c
            ValueLayout.JAVA_INT,   // c_offset
            ValueLayout.JAVA_INT,   // c_stride
            ValueLayout.JAVA_INT,   // m
            ValueLayout.JAVA_INT,   // n
            ValueLayout.JAVA_INT,   // k
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
