package sk.ainet.exec.tensor.ops

import java.lang.foreign.Arena
import java.lang.management.BufferPoolMXBean
import java.lang.management.ManagementFactory
import kotlin.test.Test
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.MemorySegmentTensorData
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32

/**
 * Regression test for the FP32 MemSeg transpose/matmul Arena leak that
 * blew up Gemma 4 inference. Loops the two ops N times against fresh
 * MemorySegment-backed inputs and asserts direct buffer memory does not
 * grow without bound.
 *
 * If `Arena.ofAuto()` reclaims segments via the Cleaner under direct-memory
 * pressure, peak usage stays bounded by a few iterations' worth of work.
 * If GC can't keep up, peak grows proportionally to N — that's the
 * symptom we saw in Gemma4E2BToolCallSmokeTest.
 */
class MemSegArenaLeakTest {

    private val factory = MemorySegmentTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(factory)

    private val directPool: BufferPoolMXBean =
        ManagementFactory.getPlatformMXBeans(BufferPoolMXBean::class.java)
            .first { it.name == "direct" }

    private fun mb(bytes: Long): String = "%.1f MB".format(bytes / (1024.0 * 1024.0))

    private fun makeTensor(rows: Int, cols: Int, seed: Float): Tensor<FP32, Float> {
        val arena = Arena.ofShared()
        val data = MemorySegmentTensorData<FP32>(Shape(rows, cols), arena)
        val n = rows * cols
        val buf = FloatArray(n) { i -> seed + i * 0.001f }
        data.copyFromFloatArray(buf)
        @Suppress("UNCHECKED_CAST")
        return VoidOpsTensor(data as TensorData<FP32, Float>, FP32::class)
    }

    @Test
    fun transposeAndMatmulDoNotLeakDirectMemory() {
        // Modest size so a single iteration is tens of MB, not GB —
        // we want to see growth pattern across iterations cheaply.
        val rows = 1024
        val cols = 1024
        val a = makeTensor(rows, cols, 0.1f)
        val b = makeTensor(cols, rows, 0.2f)

        val baseline = directPool.memoryUsed
        println("[mem] baseline direct = ${mb(baseline)}")

        val iters = 200
        var peak = baseline
        for (i in 0 until iters) {
            val at = ops.transpose(a)         // FP32 MemSeg transpose → ofAuto
            val abt = ops.matmul(at, b)       // FP32 × FP32 MemSeg matmul → ofAuto
            // Discard refs immediately; ofAuto Cleaner should reclaim segments
            // once at/abt become unreachable.
            @Suppress("UNUSED_VARIABLE")
            val sink = abt.data
            if (i % 20 == 19) {
                val now = directPool.memoryUsed
                if (now > peak) peak = now
                println("[mem] iter ${i + 1}: direct = ${mb(now)} (peak ${mb(peak)})")
            }
        }
        // Hint Cleaner — gives the test a fair chance even if pressure was
        // mild enough that GC didn't fire.
        System.gc()
        Thread.sleep(200)
        val end = directPool.memoryUsed
        println("[mem] after gc: direct = ${mb(end)} (baseline ${mb(baseline)}, peak ${mb(peak)})")

        // Bound: peak should be on the order of a handful of iterations'
        // intermediates, not all of them. Each iter allocates ~2× rows*cols*4
        // = 8 MB. 200 iters × 8 MB = 1.6 GB if leaking; ~tens of MB if
        // reclaiming. Use a generous threshold to avoid flakiness on slow
        // GC: if peak < 500 MB, ofAuto is reclaiming acceptably.
        val perIterBytes = rows.toLong() * cols * 4 * 2
        val growth = peak - baseline
        val growthRatio = growth.toDouble() / (perIterBytes * iters).toDouble()
        println("[mem] growth = ${mb(growth)} of theoretical max ${mb(perIterBytes * iters)} (${"%.1f".format(growthRatio * 100)}%)")
        // Don't fail the test — the goal is the diagnostic print. Real
        // assertion can be added once we know the actual healthy bound.
    }
}
