package sk.ainet.exec

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.ops.ValidationResult

class TracingAcceptanceTest {

    private val dataFactory = DenseTensorDataFactory()

    @Suppress("UNCHECKED_CAST")
    private fun ones(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.ones<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    // 7.1 Unit test: eager-only run yields correct numeric results and no traces/graph changes.
    @Test
    fun eagerOnly_addProducesCorrectResult_noRecordingArtifacts() {
        val ctx = DefaultGraphExecutionContext.eager(DefaultCpuOps(dataFactory))
        val ops = ctx.ops
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))

        val y = ops.add(a, b)

        // Numeric check: 1 + 1 = 2
        val v = y.data.get(0)
        assertEquals(2.0f, v, 1e-6f)

        // No tape and not recording
        assertTrue(ctx.currentTape == null)
    }

    // 7.2 Unit test: tape-only captures unary and binary ops; tape non-empty.
    @Test
    fun tapeOnly_capturesAddAndRelu_tracesNonEmpty() {
        val ctx = DefaultGraphExecutionContext.tape(DefaultCpuOps(dataFactory))
        ctx.startRecording()
        val ops = ctx.ops
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))

        val sum = ops.add(a, b)
        ops.relu(sum)

        val tape = ctx.stopRecording()
        assertNotNull(tape)
        tape as DefaultExecutionTape

        val traces = tape.traces
        assertTrue(traces.isNotEmpty(), "Tape traces should not be empty")
        val types = traces.map { it.opType }.toSet()
        assertTrue("add" in types, "Tape should contain 'add' op trace")
        assertTrue("relu" in types, "Tape should contain 'relu' op trace")
    }

    // 7.3 Unit test: graph-only builds nodes/edges for executed ops; validate passes.
    @Test
    fun graphOnly_buildsNodesAndEdges_validatePasses() {
        val graph = DefaultComputeGraph()
        val ctx = DefaultGraphExecutionContext.graph(DefaultCpuOps(dataFactory), graph)
        val ops = ctx.ops
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))

        val sum = ops.add(a, b)
        ops.relu(sum)

        // Expect two op nodes (add, relu)
        assertEquals(2, graph.nodes.size)
        // Expect one edge from add -> relu
        assertEquals(1, graph.edges.size)

        val validation = graph.validate()
        assertTrue(validation is ValidationResult.Valid, "Graph validation should pass: $validation")
    }

    // 7.4 Unit test: composite run (tape+graph) produces both, and tape→graph equals online graph modulo IDs.
    @Test
    fun composite_tapeAndGraph_offlineEqualsOnlineModuloIds() {
        val graph = DefaultComputeGraph()
        val ctx = DefaultGraphExecutionContext.tapeAndGraph(DefaultCpuOps(dataFactory), graph)

        ctx.startRecording()
        val ops = ctx.ops
        val a = ones(Shape(intArrayOf(1)))
        val b = ones(Shape(intArrayOf(1)))
        val sum = ops.add(a, b)
        ops.relu(sum)
        val tape = ctx.stopRecording() as DefaultExecutionTape

        // Online graph info
        val onlineNodes = graph.nodes.map { it.operation.name }
        val onlineEdges = graph.edges.size

        // Offline graph built from tape
        val offline = tape.toComputeGraph() as DefaultComputeGraph
        val offlineNodes = offline.nodes.map { it.operation.name }
        val offlineEdges = offline.edges.size

        assertEquals(onlineNodes, offlineNodes, "Op sequence should match modulo IDs")
        assertEquals(onlineEdges, offlineEdges, "Edge count should match")
    }

    // 7.5 Smoke test: record { add, relu } and assert node types and edge count.
    @Test
    fun smoke_recordBlock_addRelu_nodeTypesAndEdgeCount() {
        val graph = DefaultComputeGraph()
        val ctx = DefaultGraphExecutionContext.tapeAndGraph(DefaultCpuOps(dataFactory), graph)

        val (tape, _) = ctx.record {
            val ops = this.ops
            val a = ones(Shape(intArrayOf(1)))
            val b = ones(Shape(intArrayOf(1)))
            val sum = ops.add(a, b)
            ops.relu(sum)
        }
        assertNotNull(tape)

        val types = graph.nodes.map { it.operation.name }.toSet()
        assertTrue("add" in types)
        assertTrue("relu" in types)
        assertEquals(1, graph.edges.size, "Expect single edge from add to relu")
    }
}
