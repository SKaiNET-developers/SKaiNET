package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Fp32MatmulKernel

/**
 * Native (FFM) implementation of [Fp32MatmulKernel].
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_fp32_matmul(
 *       const float* a, int32_t a_offset, int32_t a_stride,
 *       const float* b, int32_t b_offset, int32_t b_stride,
 *       float* c, int32_t c_offset, int32_t c_stride,
 *       int32_t m, int32_t n, int32_t k);
 *
 * The C kernel is a tight i-p-j outer-product accumulator over rows
 * of C; the inner `c[j] += a*b[j]` loop streams two contiguous arrays
 * and auto-vectorizes into FMA under -O3 -ffast-math (vfmadd231ps on
 * x86_64, fmla on AArch64).
 *
 * Numerical parity vs [PanamaVectorMatmulKernel] is asserted by
 * [NativeFp32MatmulKernelParityTest] within FMA + reordered-reduction
 * tolerance (the same `1e-5 * k` bar Panama uses against the scalar
 * reference).
 *
 * PR 5 of the staged native-FFM rollout — wraps the rollout per the
 * `native-ffm-plan` asciidoc. Single-threaded, no cache blocking;
 * future work could add parallelChunks-style row blocking and B-tile
 * packing, but the scalar C path already lands well within the SPI
 * contract on host-arch CPUs.
 *
 * ## Per-call cost, and why callers avoid this kernel at small sizes
 *
 * The SPI hands this kernel heap `FloatArray`s, and a downcall cannot address heap memory on
 * JDK 21, so both operands are copied off-heap on every call. That copy is proportional to the
 * *weight*, not to the work: at `k=1536, n=256` it is 1.5 MB whatever `m` is. Measured on an
 * Apple M4 (`Fp32GemvShapeBench`):
 *
 * ```
 *   m= 1   1.002 ms/call   0.78 GFLOP/s        m= 8   1.075 ms/call   5.85 GFLOP/s
 *   m= 4   0.992 ms/call   3.17 GFLOP/s        m=32   1.600 ms/call  15.73 GFLOP/s
 * ```
 *
 * Eight times the arithmetic for 8% more time — below `m ~ 32` this is a fixed cost, so a decode
 * step (`m = 1`) is essentially all copy. Two things were tried and measured as no-ops, so do not
 * reach for them again: reusing the [Arena] and its segments across calls (1.025 ms vs 1.002 ms at
 * m=1 — the allocation was never the cost), and reordering the C loops (the C kernel is already
 * i-p-j). `DefaultCpuOpsJvm` therefore serves small shapes directly and leaves this kernel the
 * large ones it wins.
 *
 * Keeping the weight off-heap so it needs no copy is the obvious escape — a weight is read-only
 * for the life of the process — and it does not pay today, because the segment kernel that then
 * serves it (`JvmVectorKernels.matmulFloatBlockedMemSeg`) is slower than the heap one by more than
 * the copy costs. Measured on the same shapes, both operands `MemorySegmentTensorData`:
 *
 * ```
 *   m= 1   0.943 ms/call   0.83 GFLOP/s   (heap path: 0.122 ms, 6.44)
 *   m= 8   1.363 ms/call   4.62 GFLOP/s   (heap path: 0.770 ms, 8.17)
 *   m=32   2.785 ms/call   9.04 GFLOP/s   (heap path: 1.618 ms, 15.55)
 * ```
 *
 * So there are two independent gaps, and residency is not the lever: this kernel cannot see heap
 * memory, and the kernel that can see off-heap memory is slow. Closing either one is worth real
 * throughput — the packed Q4_K path next door reaches ~29 GFLOP/s on the same machine.
 *
 * The copy disappears on **JDK 22+**, where `Linker.Option.critical(true)` lets a downcall read
 * heap segments directly; when the toolchain moves, pass `MemorySegment.ofArray(...)` through a
 * critical handle and the small-shape threshold in `DefaultCpuOpsJvm` can be revisited.
 */
internal object NativeFp32MatmulKernel : Fp32MatmulKernel {

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        out: FloatArray, outOffset: Int, outStride: Int,
        m: Int, n: Int, k: Int,
    ) {
        require(m >= 0 && n >= 0 && k >= 0) {
            "NativeFp32MatmulKernel: m, n, k must be non-negative; got m=$m n=$n k=$k"
        }
        if (m == 0 || n == 0) return

        val mh = handle
            ?: error("NativeFp32MatmulKernel.matmul invoked while native library unavailable")

        // Sizes for the off-heap copies. Each of A, B, C uses the
        // bytes the kernel actually reaches — for non-contiguous
        // strides this can be larger than the matrix's element count
        // because the strides skip past unused floats. Allocating to
        // the full reach (offset + last-row reach) keeps the kernel
        // pointer arithmetic simple and matches Kotlin's bounds.
        val aReachFloats = if (m == 0 || k == 0) 0 else aOffset + (m - 1) * aStride + k
        val bReachFloats = if (k == 0 || n == 0) 0 else bOffset + (k - 1) * bStride + n
        val cReachFloats = outOffset + (m - 1) * outStride + n

        Arena.ofConfined().use { arena ->
            val aBytes = aReachFloats.toLong() * java.lang.Float.BYTES
            val bBytes = bReachFloats.toLong() * java.lang.Float.BYTES
            val cBytes = cReachFloats.toLong() * java.lang.Float.BYTES
            val align = ValueLayout.JAVA_FLOAT.byteAlignment()

            val aSeg: MemorySegment = if (aBytes > 0) arena.allocate(aBytes, align) else MemorySegment.NULL
            val bSeg: MemorySegment = if (bBytes > 0) arena.allocate(bBytes, align) else MemorySegment.NULL
            val cSeg: MemorySegment = arena.allocate(cBytes, align)

            if (aReachFloats > 0) {
                MemorySegment.copy(a, 0, aSeg, ValueLayout.JAVA_FLOAT, 0L, aReachFloats)
            }
            if (bReachFloats > 0) {
                MemorySegment.copy(b, 0, bSeg, ValueLayout.JAVA_FLOAT, 0L, bReachFloats)
            }

            mh.invoke(
                aSeg, aOffset, aStride,
                bSeg, bOffset, bStride,
                cSeg, outOffset, outStride,
                m, n, k,
            )

            MemorySegment.copy(cSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, cReachFloats)
        }
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_fp32_matmul").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // a
            ValueLayout.JAVA_INT,   // a_offset
            ValueLayout.JAVA_INT,   // a_stride
            ValueLayout.ADDRESS,    // b
            ValueLayout.JAVA_INT,   // b_offset
            ValueLayout.JAVA_INT,   // b_stride
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
