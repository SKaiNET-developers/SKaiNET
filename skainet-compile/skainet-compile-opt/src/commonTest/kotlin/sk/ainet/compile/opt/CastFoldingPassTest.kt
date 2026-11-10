package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.CastFoldingPass
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class CastFoldingPassTest {

    private fun spec(dtype: String) = TensorSpec(name = "t", shape = listOf(1, 4), dtype = dtype)

    private fun node(id: String, op: String, inDtype: String?, outDtype: String) = GraphNode(
        id = id,
        operation = GenericOperation(op),
        inputs = listOfNotNull(inDtype?.let { spec(it) }),
        outputs = listOf(spec(outDtype)),
    )

    /** A chain `x -> cast -> cast -> … -> relu`, one cast per consecutive dtype pair. */
    private fun chain(vararg dtypes: String, castOp: String = "convert"): ComputeGraph {
        val graph = DefaultComputeGraph()
        var previous = graph.addNode(node("x", "input", null, dtypes.first()))
        for (i in 1 until dtypes.size) {
            val cast = graph.addNode(node("c$i", castOp, dtypes[i - 1], dtypes[i]))
            graph.addEdge(GraphEdge("e$i", previous, cast, tensorSpec = spec(dtypes[i - 1])))
            previous = cast
        }
        val sink = graph.addNode(node("out", "relu", dtypes.last(), dtypes.last()))
        graph.addEdge(GraphEdge("eOut", previous, sink, tensorSpec = spec(dtypes.last())))
        return graph
    }

    /** The casts of [graph] in execution order, as `from->to`, read off the edges. */
    private fun casts(graph: ComputeGraph): List<String> =
        graph.getTopologicalOrder()
            .filter { it.operationName in setOf("convert", "cast", "to") }
            .map { cast ->
                val edge = graph.edges.single { it.destination.id == cast.id }
                assertEquals(edge.tensorSpec.dtype, cast.inputs.single().dtype, "input spec of ${cast.id} must match its edge")
                "${edge.source.outputs[edge.sourceOutputIndex].dtype}->${cast.outputs.single().dtype}"
            }

    private fun fold(graph: ComputeGraph): GraphOptimizationResult = CastFoldingPass().apply(graph)

    @Test
    fun wideningThenSameNarrowDtype_removesBothCasts() {
        val result = fold(chain("FP16", "FP32", "FP16"))

        assertTrue(result.changed)
        assertTrue(casts(result.graph).isEmpty())
        val sinkEdge = result.graph.edges.single()
        assertEquals("x", sinkEdge.source.id)
        assertEquals("out", sinkEdge.destination.id)
        assertEquals("FP16", sinkEdge.tensorSpec.dtype)
        assertEquals(ValidationResult.Valid, result.graph.validate())
    }

    @Test
    fun wideningThenOtherNarrowDtype_becomesOneCast() {
        val result = fold(chain("BF16", "FP32", "FP16"))

        assertTrue(result.changed)
        assertEquals(listOf("BF16->FP16"), casts(result.graph))
        assertEquals(setOf("x", "c2", "out"), result.graph.nodes.map { it.id }.toSet())
    }

    @Test
    fun adjacentBoundaries_keepTheirNarrowingAndLoseTheRoundTrip() {
        // FP32 network, BF16 layer directly followed by an FP16 stage.
        val result = fold(chain("FP32", "BF16", "FP32", "FP16", "FP32"))

        assertEquals(listOf("FP32->BF16", "BF16->FP16", "FP16->FP32"), casts(result.graph))
    }

    @Test
    fun narrowingRoundTrip_isNeverFolded() {
        // f32 -> bf16 -> f32 is the rounding a mixed-precision layer asked for.
        val graph = chain("FP32", "BF16", "FP32")

        val result = fold(graph)

        assertFalse(result.changed)
        assertSame(graph, result.graph)
    }

    @Test
    fun narrowToNarrow_isNotAnExactWidening() {
        val graph = chain("FP16", "BF16", "FP16")
        assertFalse(fold(graph).changed)
    }

    @Test
    fun integerCasts_areNotTreatedAsExact() {
        // Int32 -> FP32 loses precision above 2^24.
        val graph = chain("Int32", "FP32", "Int32")
        assertFalse(fold(graph).changed)
    }

    @Test
    fun castToTheSameDtype_isRemoved() {
        val result = fold(chain("BF16", "BF16"))

        assertTrue(result.changed)
        assertTrue(casts(result.graph).isEmpty())
        assertEquals("x", result.graph.edges.single().source.id)
    }

    @Test
    fun chainOfWidenings_collapses() {
        val result = fold(chain("FP16", "FP32", "FP64", "FP16"))
        assertTrue(casts(result.graph).isEmpty())
    }

    @Test
    fun wideningWithAnotherConsumer_staysAndTheSecondCastReadsTheSource() {
        // x(BF16) -> widen(FP32) -> narrow(FP16) -> out
        //                       \-> other (reads the FP32 tensor)
        val graph = DefaultComputeGraph()
        val x = graph.addNode(node("x", "input", null, "BF16"))
        val widen = graph.addNode(node("widen", "convert", "BF16", "FP32"))
        val narrow = graph.addNode(node("narrow", "convert", "FP32", "FP16"))
        val out = graph.addNode(node("out", "relu", "FP16", "FP16"))
        val other = graph.addNode(node("other", "relu", "FP32", "FP32"))
        graph.addEdge(GraphEdge("e1", x, widen, tensorSpec = spec("BF16")))
        graph.addEdge(GraphEdge("e2", widen, narrow, tensorSpec = spec("FP32")))
        graph.addEdge(GraphEdge("e3", narrow, out, tensorSpec = spec("FP16")))
        graph.addEdge(GraphEdge("e4", widen, other, tensorSpec = spec("FP32")))

        val result = fold(graph)

        assertTrue(result.changed)
        assertEquals(setOf("BF16->FP32", "BF16->FP16"), casts(result.graph).toSet())
        assertEquals("widen", result.graph.edges.single { it.destination.id == "other" }.source.id)
        assertEquals("x", result.graph.edges.single { it.destination.id == "narrow" }.source.id)
    }

    @Test
    fun castAliasesAndDtypeSpellings_areRecognised() {
        val result = fold(chain("bf16", "float32", "f16", castOp = "cast"))
        assertEquals(1, result.graph.nodes.count { it.operationName == "cast" })
    }

    @Test
    fun graphWithoutCasts_isReturnedUnchanged() {
        val graph = DefaultComputeGraph()
        val x = graph.addNode(node("x", "input", null, "FP32"))
        val out = graph.addNode(node("out", "relu", "FP32", "FP32"))
        graph.addEdge(GraphEdge("e", x, out, tensorSpec = spec("FP32")))

        val result = fold(graph)

        assertFalse(result.changed)
        assertSame(graph, result.graph)
    }

    @Test
    fun diagnosticsNameEveryRewrite() {
        val result = fold(chain("FP32", "BF16", "FP32", "FP16", "FP32"))
        assertEquals(1, result.diagnostics.size, result.diagnostics.toString())
        assertTrue("BF16 -> FP16" in result.diagnostics.single(), result.diagnostics.single())
    }
}
