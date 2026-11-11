package sk.ainet.exec.tensor.ops

import java.lang.management.BufferPoolMXBean
import java.lang.management.ManagementFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.forwardScope
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32

/**
 * SKEEP-003 acceptance criterion (#1501): repeated forward passes under a forward scope keep
 * memory flat on the JVM. This is the failure that forced the retreat to the garbage collector:
 * a shared arena grew with every op output, and per-call arenas leaked per matmul.
 *
 * The loop is a decode step in miniature — weights outside the scope, activations inside it,
 * `reset()` at the step boundary — at a size where a leak is unmistakable: one step produces
 * about 3 MB of activations, so 300 steps would be about 900 MB if anything accumulated.
 * Asserted: the direct buffer pool, the live heap after a collection, and the slab's peak.
 */
class ForwardScopeFlatMemoryTest {

    private val n = 512
    private val steps = 300
    private val warmup = 20

    private val directPool: BufferPoolMXBean =
        ManagementFactory.getPlatformMXBeans(BufferPoolMXBean::class.java).first { it.name == "direct" }

    /** Live heap after a collection: what the loop still holds, not what it has churned through. */
    private fun liveHeap(): Long {
        val rt = Runtime.getRuntime()
        repeat(3) { System.gc(); Thread.sleep(50) }
        return rt.totalMemory() - rt.freeMemory()
    }

    private fun mb(bytes: Long): String = "%.1f MB".format(bytes / (1024.0 * 1024.0))

    private fun <V> step(x: Tensor<FP32, V>, w: Tensor<FP32, V>, b: Tensor<FP32, V>): Tensor<FP32, V> {
        val ops = x.ops
        return ops.relu(ops.add(ops.matmul(x, w), b))
    }

    private fun weights(ctx: ExecutionContext): Pair<Tensor<FP32, Float>, Tensor<FP32, Float>> {
        val w = ctx.fromFloatArray<FP32, Float>(Shape(n, n), FP32::class, FloatArray(n * n) { ((it * 31) % 17 - 8) * 0.01f })
        val b = ctx.fromFloatArray<FP32, Float>(Shape(n), FP32::class, FloatArray(n) { (it % 5 - 2) * 0.1f })
        return w to b
    }

    @Test
    fun repeatedForwardPassesKeepDirectMemoryAndTheSlabFlat() {
        val base = DirectCpuExecutionContext()
        val (w, b) = weights(base)
        val input = FloatArray(n * n) { ((it * 13) % 11 - 5) * 0.05f }
        val stepBytes = 3L * n * n * Float.SIZE_BYTES // input, matmul and add outputs; relu reuses or adds one more

        base.forwardScope(slabFloats = 8 * n * n) { scoped, scope ->
            var directAfterWarmup = 0L
            var rssAfterWarmup: Long? = null
            var heapAfterWarmup = 0L
            var peakAfterWarmup = 0
            var directPeak = 0L

            repeat(steps) { stepNo ->
                val x = scoped.fromFloatArray<FP32, Float>(Shape(n, n), FP32::class, input)
                val y = step(x, w, b)
                assertTrue(y.data is StorageFloatTensorData<*>, "step $stepNo: activations must come from the slab")
                if (stepNo == 0) assertTrue(y.data.copyToFloatArray().any { it > 0f }, "the step must compute something")
                assertEquals(0L, scope.overflowBytes, "step $stepNo: nothing may spill past the slab")
                scope.reset()

                val direct = directPool.memoryUsed
                if (stepNo == warmup) {
                    directAfterWarmup = direct
                    rssAfterWarmup = MemoryProbe.rssBytes()
                    heapAfterWarmup = liveHeap()
                    peakAfterWarmup = scope.peakFloats
                }
                if (stepNo > warmup) {
                    directPeak = maxOf(directPeak, direct)
                    assertEquals(peakAfterWarmup, scope.peakFloats, "step $stepNo: the slab must not grow in steady state")
                }
            }

            val directGrowth = directPeak - directAfterWarmup
            val heapGrowth = liveHeap() - heapAfterWarmup
            val rssGrowth = rssAfterWarmup?.let { before -> MemoryProbe.rssBytes()?.minus(before) }
            println(
                "[forward-scope] ${steps - warmup} steady steps of ${mb(stepBytes)}: direct pool ${mb(directAfterWarmup)} -> peak ${mb(directPeak)}, " +
                    "live heap growth ${mb(heapGrowth)}, rss growth ${rssGrowth?.let(::mb) ?: "n/a"}, slab peak ${scope.peakFloats} floats",
            )

            // Flat: less than one step's worth of direct memory appears over 280 steps.
            assertTrue(directGrowth < stepBytes, "direct memory grew by ${mb(directGrowth)} over ${steps - warmup} steps")
            // A leak of every activation would be about 840 MB here. Live heap after a collection
            // is what the loop still holds; allow a tenth of the leak for whatever else the test
            // JVM keeps. RSS is printed, not asserted: a JVM grows its committed heap lazily, so
            // RSS moves with garbage that is already dead (seen on a CI runner).
            val leakIfEverythingStayed = stepBytes * (steps - warmup)
            assertTrue(heapGrowth < leakIfEverythingStayed / 10, "live heap grew by ${mb(heapGrowth)} over ${steps - warmup} steps")
        }
    }
}
