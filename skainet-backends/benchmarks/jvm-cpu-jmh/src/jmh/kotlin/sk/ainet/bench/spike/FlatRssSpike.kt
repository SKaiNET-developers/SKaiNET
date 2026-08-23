package sk.ainet.bench.spike

import java.io.File
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout

/**
 * SKEEP-003 Phase-2 spike, second half (decision #6, #1016): is decode-step memory flat when
 * activations come from a recycled `Forward` slab (`Arena.ofShared` + bump offset + `reset()`
 * per step) — versus fresh heap arrays per step (today's behaviour, GC-backed)?
 *
 * Simulates a 1B-class decode step: per step `LAYERS × K` activation buffers of `HIDDEN` floats
 * (plus a `VOCAB` logits row), written and read once. Prints RSS (from /proc/self/status) at
 * steps 50, 500, 1000, … and the allocation count. Run:
 *
 * `./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:runFlatRssSpike` (see build.gradle.kts).
 */
object FlatRssSpike {
    private const val LAYERS = 16
    private const val BUFFERS_PER_LAYER = 12
    private const val HIDDEN = 2048
    private const val VOCAB = 128_256

    /** The Forward-slab model: one shared arena, bump allocation, reset per step — zero steady-state allocation. */
    class ForwardSlab(bytes: Long) {
        private val arena = Arena.ofShared()
        private val slab: MemorySegment = arena.allocate(bytes, 64)
        private var offset = 0L
        var allocations = 0L; private set
        fun allocate(bytes: Long): MemorySegment {
            val aligned = (bytes + 63) and 63L.inv()
            require(offset + aligned <= slab.byteSize()) { "Forward slab exhausted: need $aligned at $offset of ${slab.byteSize()}" }
            val s = slab.asSlice(offset, bytes); offset += aligned; allocations++; return s
        }
        fun reset() { offset = 0L }
        fun close() = arena.close()
    }

    private fun rssMiB(): Long = File("/proc/self/status").readLines().firstOrNull { it.startsWith("VmRSS:") }
        ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.let { it / 1024 } ?: -1

    @JvmStatic
    fun main(args: Array<String>) {
        val steps = args.getOrNull(0)?.toIntOrNull() ?: 2000
        val mode = args.getOrNull(1) ?: "both"
        val stepBytes = LAYERS.toLong() * BUFFERS_PER_LAYER * HIDDEN * 4 + VOCAB.toLong() * 4
        println("FlatRssSpike: $steps steps, per-step activations = ${stepBytes / 1024 / 1024} MiB, pid=${ProcessHandle.current().pid()}")
        if (mode == "both" || mode == "slab") runSlab(steps, stepBytes)
        if (mode == "both" || mode == "heap") runHeap(steps)
    }

    private fun checkpoints(steps: Int) = setOf(1, 50, 100, 500, 1000, 1500, 2000, 3000, 5000, steps)

    private fun heapUsedMiB(): Long = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024

    private fun runSlab(steps: Int, stepBytes: Long) {
        val slab = ForwardSlab(stepBytes + (1 shl 20))
        val marks = checkpoints(steps) // hoisted: the harness itself must not allocate per step
        var sink = 0f
        println("--- Forward slab (Arena.ofShared bump + reset per step) ---")
        for (step in 1..steps) {
            for (l in 0 until LAYERS) for (b in 0 until BUFFERS_PER_LAYER) {
                val seg = slab.allocate(HIDDEN * 4L)
                seg.setAtIndex(ValueLayout.JAVA_FLOAT, (step % HIDDEN).toLong(), step.toFloat())
                sink += seg.getAtIndex(ValueLayout.JAVA_FLOAT, (step % HIDDEN).toLong())
            }
            val logits = slab.allocate(VOCAB * 4L); logits.setAtIndex(ValueLayout.JAVA_FLOAT, 0, sink)
            slab.reset()
            if (step in marks) println("  step %5d  RSS %6d MiB  heapUsed %4d MiB  slab allocations so far %d (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)".format(step, rssMiB(), heapUsedMiB(), slab.allocations))
        }
        slab.close()
        println("  sink=$sink")
    }

    private fun runHeap(steps: Int) {
        var sink = 0f
        var allocations = 0L
        val marks = checkpoints(steps)
        println("--- Heap arrays per step (today: GC-backed) ---")
        for (step in 1..steps) {
            for (l in 0 until LAYERS) for (b in 0 until BUFFERS_PER_LAYER) {
                val arr = FloatArray(HIDDEN); allocations++
                arr[step % HIDDEN] = step.toFloat(); sink += arr[step % HIDDEN]
            }
            val logits = FloatArray(VOCAB); allocations++; logits[0] = sink
            if (step in marks) println("  step %5d  RSS %6d MiB  heapUsed %4d MiB  heap allocations so far %d".format(step, rssMiB(), heapUsedMiB(), allocations))
        }
        println("  sink=$sink")
    }
}
