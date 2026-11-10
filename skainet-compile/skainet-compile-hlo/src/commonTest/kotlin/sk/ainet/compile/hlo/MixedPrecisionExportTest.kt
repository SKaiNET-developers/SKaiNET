package sk.ainet.compile.hlo

import sk.ainet.compile.opt.passes.CastFoldingPass
import sk.ainet.compile.opt.passes.DtypeForwardPropagationPass
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.network
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * SKEEP-008 phase A2 (#1433): a network with `dense(dtype = …)` / `stage(id, dtype = …)`
 * traces with a `convert` node on each side of every [sk.ainet.lang.nn.PrecisionBoundary],
 * and exports to StableHLO with one `stablehlo.convert` pair per boundary.
 */
class MixedPrecisionExportTest {

    /** Traces one forward pass of the network [build] makes, on the CPU ops. */
    private fun trace(
        build: (DefaultGraphExecutionContext) -> Module<FP32, Float>,
    ): ComputeGraph {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = DirectCpuExecutionContext().ops)
        val model = build(ctx)
        val input = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 4), dtype = FP32::class,
            data = floatArrayOf(0.3f, -1.1f, 0.7f, 2.5f),
        )
        val inputRefId = ctx.session.refOf(input as Tensor<*, *>).id
        val (tape, result) = ctx.record { model.forward(input, ctx) }
        assertNotNull(tape, "the forward pass must be recorded")
        assertEquals(FP32::class, result?.dtype, "the last boundary must cast back to the network dtype")
        return tape.toComputeGraph(synthesizeExternalInputs = true, inputTensorIds = setOf(inputRefId))
    }

    /** FP32 network with one BF16 dense and one FP16 stage: the SKEEP-008 acceptance network. */
    private fun mixedNetwork(ctx: DefaultGraphExecutionContext): Module<FP32, Float> = definition {
        network<FP32, Float>(ctx) {
            input(4)
            dense(3, dtype = BF16, id = "bf16Dense")
            stage("head", dtype = FP16) {
                dense(3, "h1")
                dense(2, "h2")
            }
        }
    }

    private fun convertLines(mlir: String): List<String> =
        mlir.lines().map { it.trim() }.filter { "stablehlo.convert" in it }

    @Test
    fun trace_hasOneConvertPairPerBoundary() {
        val graph = trace(::mixedNetwork)

        val casts = graph.getTopologicalOrder()
            .filter { it.operation.name == "convert" }
            .map { it.inputs.single().dtype to it.outputs.single().dtype }

        assertEquals(4, casts.size, "two boundaries, one cast in and one cast out each: $casts")
        assertEquals(
            setOf("FP32" to "BF16", "BF16" to "FP32", "FP32" to "FP16", "FP16" to "FP32"),
            casts.toSet(),
        )
    }

    @Test
    fun trace_runsInnerLayersInTheNarrowDtype() {
        val graph = trace(::mixedNetwork)

        val matmulDtypes = graph.getTopologicalOrder()
            .filter { it.operation.name == "matmul" }
            .map { it.outputs.single().dtype }

        // One BF16 dense, then the two dense layers of the FP16 stage. No FP32 matmul:
        // every layer of this network sits behind a boundary.
        assertEquals(listOf("BF16", "FP16", "FP16"), matmulDtypes)
    }

    @Test
    fun export_emitsStableHloConvertAtEachBoundary_inOrder() {
        val module = StableHloConverterFactory.createExtended().convert(trace(::mixedNetwork), "mixed")
        val mlir = module.content

        assertFalse("Unsupported" in mlir, "every op must lower:\n$mlir")
        assertFalse("tensor<?" in mlir, "shapes must stay static:\n$mlir")

        val signatures = convertLines(mlir).map { it.substringAfter(" : ") }
        assertEquals(
            listOf(
                "(tensor<1x4xf32>) -> tensor<1x4xbf16>",
                "(tensor<1x3xbf16>) -> tensor<1x3xf32>",
                "(tensor<1x3xf32>) -> tensor<1x3xf16>",
                "(tensor<1x2xf16>) -> tensor<1x2xf32>",
            ),
            signatures,
            "one stablehlo.convert pair per boundary, in execution order:\n$mlir",
        )

        // Inner computation is typed in the narrow dtype; the function signature stays FP32.
        assertTrue("-> tensor<1x3xbf16>" in mlir.lines().first { "stablehlo.dot_general" in it }, mlir)
        assertEquals(
            2,
            mlir.lines().count { "stablehlo.dot_general" in it && it.trim().endsWith("xf16>") },
            "both dense layers of the FP16 stage must be f16:\n$mlir",
        )
        assertTrue("(%arg0: tensor<1x4xf32>) -> (tensor<1x2xf32>)" in mlir, mlir)
    }

    @Test
    fun export_sameDtypeBoundary_emitsNoConvert() {
        // convert returns its input when the dtype already matches, so nothing is recorded:
        // an FP32 stage inside an FP32 network costs nothing in the exported module either.
        val graph = trace { ctx ->
            definition {
                network<FP32, Float>(ctx) {
                    input(4)
                    stage("body", dtype = FP32) {
                        dense(3, "h1")
                        dense(2, "h2")
                    }
                }
            }
        }

        assertTrue(graph.getTopologicalOrder().none { it.operation.name == "convert" })
        val mlir = StableHloConverterFactory.createExtended().convert(graph, "same").content
        assertTrue(convertLines(mlir).isEmpty(), "no cast expected:\n$mlir")
    }

    @Test
    fun castFolding_dropsTheRoundTripBetweenAdjacentBoundaries() {
        // The BF16 dense is directly followed by the FP16 stage: bf16 -> f32 -> f16 is one
        // exact widening followed by a narrowing, and folds to bf16 -> f16 (#1494).
        val folded = CastFoldingPass().apply(trace(::mixedNetwork))
        assertTrue(folded.changed)

        val mlir = StableHloConverterFactory.createExtended().convert(folded.graph, "mixed").content

        assertFalse("Unsupported" in mlir, mlir)
        assertEquals(
            listOf(
                "(tensor<1x4xf32>) -> tensor<1x4xbf16>",
                "(tensor<1x3xbf16>) -> tensor<1x3xf16>",
                "(tensor<1x2xf16>) -> tensor<1x2xf32>",
            ),
            convertLines(mlir).map { it.substringAfter(" : ") },
            mlir,
        )
    }

    @Test
    fun castFolding_givesBitIdenticalResults() {
        // The cast chain two adjacent boundaries produce, on values no 16-bit format holds:
        // executing the folded graph must reproduce the unfolded one bit for bit.
        fun cast(id: String, from: String, to: String) = GraphNode(
            id = id,
            operation = GenericOperation("convert"),
            inputs = listOf(TensorSpec("t", listOf(1, 6), from)),
            outputs = listOf(TensorSpec("t", listOf(1, 6), to)),
        )
        val graph = DefaultComputeGraph()
        var previous = graph.addNode(
            GraphNode("x", GenericOperation("input"), emptyList(), listOf(TensorSpec("x", listOf(1, 6), "FP32")))
        )
        val steps = listOf("FP32", "BF16", "FP32", "FP16", "FP32")
        for (i in 1 until steps.size) {
            val node = graph.addNode(cast("c$i", steps[i - 1], steps[i]))
            graph.addEdge(GraphEdge("e$i", previous, node, tensorSpec = TensorSpec("t", listOf(1, 6), steps[i - 1])))
            previous = node
        }
        val folded = CastFoldingPass().apply(graph).graph
        assertEquals(3, folded.nodes.count { it.operationName == "convert" })

        val cpu = DirectCpuExecutionContext()
        val values = floatArrayOf(0.1234567f, -1.7654321f, 2.3456789f, 65000.5f, 1.0e-6f, -3.3333333f)
        val x = cpu.fromFloatArray<FP32, Float>(Shape(1, 6), FP32::class, values)

        fun run(g: ComputeGraph): List<Int> {
            val out = ComputeGraphExecutor(g, cpu.ops).execute(mapOf("x" to x)).values.single()
            return List(6) { out.data[0, it].toRawBits() }
        }

        val expected = run(graph)
        assertEquals(expected, run(folded))
        assertTrue(expected != values.map { it.toRawBits() }, "the chain must round, or it proves nothing")
    }

    @Test
    fun dtypePropagation_keepsTheBoundaries() {
        // The pass makes edges dtype-consistent and must treat convert as intentionally
        // dtype-changing: a traced mixed-precision graph comes out with the same casts
        // and exports to the same module.
        val graph = trace(::mixedNetwork)
        val propagated = DtypeForwardPropagationPass().apply(graph).graph

        val converter = StableHloConverterFactory.createExtended()
        assertEquals(
            convertLines(converter.convert(graph, "mixed").content),
            convertLines(converter.convert(propagated, "mixed").content),
        )
    }
}
