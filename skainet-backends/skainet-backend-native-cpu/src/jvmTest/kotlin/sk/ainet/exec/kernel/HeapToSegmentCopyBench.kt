package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * How fast is heap `FloatArray` → off-heap `MemorySegment`, really?
 *
 * `NativeFp32MatmulKernel` copies both operands off-heap on every call because a JDK 21 downcall
 * cannot address heap memory. Decomposing its measured cost (1.002 ms at m=1, 1.600 ms at m=32,
 * k=1536 n=256) as `total = copy + compute·m` puts compute at ~19 µs — the C kernel is doing
 * ~41 GFLOP/s — and the copy at ~0.98 ms for 1.5 MB, i.e. ~1.5 GB/s. That is an order of magnitude
 * under this machine's memory bandwidth, so the question is whether the copy is intrinsified at all
 * or is quietly running element-wise.
 *
 * Compares the spellings available on JDK 21 for the same 1.5 MB.
 */
class HeapToSegmentCopyBench {

    @Test
    fun heap_to_offheap_copy_throughput() {
        if (System.getenv("SKAINET_BENCH") != "1") {
            println("[skip] set SKAINET_BENCH=1 to run the copy benchmark"); return
        }
        val floats = 1536 * 256                      // the Gemma 4 per-layer-embedding weight
        val bytes = floats.toLong() * Float.SIZE_BYTES
        val src = FloatArray(floats) { it * 0.001f }
        val iterations = 500

        fun report(label: String, elapsedMs: Double) {
            val gbPerSec = bytes.toDouble() * iterations / (elapsedMs / 1000.0) / 1e9
            println("COPY %-42s %7.3f ms/copy  %6.2f GB/s".format(label, elapsedMs / iterations, gbPerSec))
        }

        Arena.ofConfined().use { arena ->
            val dst = arena.allocate(bytes, ValueLayout.JAVA_FLOAT.byteAlignment())

            // 1. What the kernel does today.
            repeat(50) { MemorySegment.copy(src, 0, dst, ValueLayout.JAVA_FLOAT, 0L, floats) }
            report("MemorySegment.copy(JAVA_FLOAT)", measureTime {
                repeat(iterations) { MemorySegment.copy(src, 0, dst, ValueLayout.JAVA_FLOAT, 0L, floats) }
            }.inWholeMicroseconds / 1000.0)

            // 2. Same, with the layout's byte order pinned to native. An unaligned or non-native
            //    layout is the usual reason this drops off its intrinsic.
            val nativeLayout = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.nativeOrder())
            repeat(50) { MemorySegment.copy(src, 0, dst, nativeLayout, 0L, floats) }
            report("MemorySegment.copy(JAVA_FLOAT native order)", measureTime {
                repeat(iterations) { MemorySegment.copy(src, 0, dst, nativeLayout, 0L, floats) }
            }.inWholeMicroseconds / 1000.0)

            // 3. Segment-to-segment bulk copy, wrapping the heap array as a segment. This is the
            //    memcpy-shaped spelling and does not go through a ValueLayout at all.
            val srcSeg = MemorySegment.ofArray(src)
            repeat(50) { dst.copyFrom(srcSeg) }
            report("dst.copyFrom(MemorySegment.ofArray(src))", measureTime {
                repeat(iterations) { dst.copyFrom(srcSeg) }
            }.inWholeMicroseconds / 1000.0)

            // 4. Floor: heap-to-heap arraycopy, for scale.
            val heapDst = FloatArray(floats)
            repeat(50) { System.arraycopy(src, 0, heapDst, 0, floats) }
            report("System.arraycopy (heap->heap, for scale)", measureTime {
                repeat(iterations) { System.arraycopy(src, 0, heapDst, 0, floats) }
            }.inWholeMicroseconds / 1000.0)
        }
    }
}
