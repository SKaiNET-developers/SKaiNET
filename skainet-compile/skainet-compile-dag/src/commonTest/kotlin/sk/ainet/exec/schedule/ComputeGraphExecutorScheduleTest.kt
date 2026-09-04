package sk.ainet.exec.schedule

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * SKEEP-005 phase 2: the compiled JVM leg runs under the schedule of the ops it was built with.
 * `ComputeGraphExecutor` dispatches an sdpa node to `ops.scaledDotProductAttention`, so a graph
 * executed with a scheduled context's ops routes its (batch, head) units through that schedule
 * and produces the sequential result bit for bit.
 */
class ComputeGraphExecutorScheduleTest {

    /** Records every range, hands them out in reverse order, and runs them inline. */
    private class ReversingSchedule(override val parallelism: Int) : Schedule {
        val ranges = mutableListOf<Pair<Int, Int>>()
        override val name: String get() = "reversing($parallelism)"
        override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
            val tasks = Schedule.tasksFor(n, grain, parallelism)
            if (tasks == 0) return
            val chunk = Schedule.chunkFor(n, tasks)
            val bounds = (0 until n step chunk).map { s -> s to minOf(s + chunk, n) }
            for ((s, e) in bounds.reversed()) { ranges += s to e; body(s, e) }
        }
    }

    // batch 1 × 8 heads × 64 queries × 64 keys × 64 dims × 2 ≈ 4.2 M multiply-adds: above SDPA_PARALLEL_MIN_WORK.
    private val heads = 8; private val seq = 64; private val headDim = 64

    private fun sdpaGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        fun input(id: String) = GraphNode(id, InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec(id, listOf(1, heads, seq, headDim), "FP32")))
        val q = input("q"); val k = input("k"); val v = input("v")
        val sdpa = GraphNode(
            "sdpa", ScaledDotProductAttentionOperation(mapOf("scale" to 0.125f, "causal" to true)),
            listOf(q.outputs[0], k.outputs[0], v.outputs[0]), listOf(TensorSpec("out", listOf(1, heads, seq, headDim), "FP32")),
        )
        listOf(q, k, v, sdpa).forEach(graph::addNode)
        graph.addEdge(GraphEdge("eq", q, sdpa, 0, 0, q.outputs[0]))
        graph.addEdge(GraphEdge("ek", k, sdpa, 0, 1, k.outputs[0]))
        graph.addEdge(GraphEdge("ev", v, sdpa, 0, 2, v.outputs[0]))
        return graph
    }

    private fun run(ctx: DirectCpuExecutionContext): FloatArray {
        fun fill(seed: Int) = FloatArray(heads * seq * headDim) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }
        val shape = Shape(1, heads, seq, headDim)
        val inputs: Map<String, Tensor<FP32, Float>> = mapOf(
            "q" to ctx.fromFloatArray(shape, FP32::class, fill(1)),
            "k" to ctx.fromFloatArray(shape, FP32::class, fill(2)),
            "v" to ctx.fromFloatArray(shape, FP32::class, fill(3)),
        )
        val outputs = ComputeGraphExecutor(sdpaGraph(), ctx.ops).execute(inputs)
        return outputs.getValue("sdpa").data.copyToFloatArray()
    }

    @Test
    fun executorRoutesSdpaThroughTheOpsSchedule() {
        val probe = ReversingSchedule(parallelism = 3)
        val scheduled = run(DirectCpuExecutionContext(schedule = probe))
        assertTrue(probe.ranges.isNotEmpty(), "the sdpa node must run through the schedule of the ops it was built with")
        assertTrue(probe.ranges.sumOf { (s, e) -> e - s } == heads, "every (batch, head) unit exactly once: ${probe.ranges}")
        assertContentEquals(run(DirectCpuExecutionContext(schedule = Schedule.Sequential)), scheduled)
    }
}
