package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * End-to-end test: tape-record a conv1d → gelu → add pipeline,
 * convert to ComputeGraph, export to StableHLO MLIR, and verify
 * the output is valid for iree-compile.
 *
 * This test covers two upstream issues:
 *   A) Conv1dOperation.inferOutputs() must compute the correct output shape
 *      (not echo the input shape), so the MLIR has static tensor types.
 *   B) toComputeGraph() must wire edges correctly so all ops have the
 *      expected number of inputs and a recognized operation type.
 *
 * Place in: skainet-compile/skainet-compile-hlo/src/commonTest/kotlin/sk/ainet/compile/hlo/
 * Run:      ./gradlew :skainet-compile:skainet-compile-hlo:allTests --tests "*Conv1dTapeToHloTest*"
 *
 * CURRENTLY FAILS — will pass once both issues are fixed.
 */
class Conv1dTapeToHloTest {

    /**
     * Record conv1d(input, weight, bias) → gelu → add(residual)
     * through VoidTensorOps, then export to StableHLO.
     *
     * Shapes:
     *   input:   [1, 3, 16]   (batch=1, channels=3, length=16)
     *   weight:  [8, 3, 3]    (out_channels=8, in_channels=3, kernel=3)
     *   bias:    [8]
     *   conv out: [1, 8, 16]  (stride=1, padding=1)
     *   gelu out: [1, 8, 16]
     *   add out:  [1, 8, 16]  (residual connection — requires a second input)
     */
    @Test
    fun conv1d_gelu_add_produces_valid_stablehlo() {
        // 1. Create tape-recording context
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        // 2. Create tensors (values irrelevant — VoidTensorOps only tracks shapes)
        val input = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 3, 16), dtype = FP32::class,
            data = FloatArray(1 * 3 * 16)
        )
        val weight = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8, 3, 3), dtype = FP32::class,
            data = FloatArray(8 * 3 * 3)
        )
        val bias = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8), dtype = FP32::class,
            data = FloatArray(8)
        )
        // A second input for the residual add — same shape as conv output
        val residual = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 8, 16), dtype = FP32::class,
            data = FloatArray(1 * 8 * 16)
        )

        // Mark the input tensor as a function argument (not a constant)
        @Suppress("UNCHECKED_CAST")
        val inputRefId = ctx.session.refOf(input as sk.ainet.lang.tensor.Tensor<*, *>).id

        // 3. Tape-record the forward pass
        val (tape, result) = ctx.record {
            val convOut = ctx.ops.conv1d(input, weight, bias, stride = 1, padding = 1)
            val geluOut = ctx.ops.gelu(convOut)
            ctx.ops.add(geluOut, residual)
        }

        assertNotNull(tape, "Tape should not be null after recording")
        assertNotNull(result, "Result tensor should not be null")

        // 4. Convert tape to compute graph
        val graph = tape.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRefId)
        )

        val nodes = graph.getTopologicalOrder()
        assertTrue(nodes.isNotEmpty(), "Graph should have nodes")
        println("Graph: ${nodes.size} nodes")

        val opCounts = nodes.groupBy { it.operation.name }.mapValues { it.value.size }
        println("Ops: $opCounts")

        // 5. Convert to StableHLO
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "conv1d_gelu_add")

        assertNotNull(module, "StableHLO module should not be null")
        println("MLIR:\n${module.content}")

        // ===== ASSERTIONS THAT COVER BOTH ISSUES =====

        // Issue A: conv1d output shape must be static (not tensor<?xf32>)
        assertFalse(
            module.content.contains("tensor<?"),
            "MLIR must not contain dynamic shapes (tensor<?...>). " +
            "Conv1dOperation.inferOutputs() should compute [1, 8, 16], not echo input shape."
        )

        // Issue B: all ops must be recognized (no "Unsupported" comments)
        assertFalse(
            module.content.contains("Unsupported"),
            "MLIR must not contain 'Unsupported' ops. " +
            "toComputeGraph() must wire edges correctly so add has 2 inputs, " +
            "gelu has 1 input, and conv1d has 2-3 inputs."
        )

        // Positive: valid StableHLO ops must be present
        assertContains(module.content, "stablehlo.convolution",
            message = "MLIR must contain stablehlo.convolution for the conv1d op")

        assertContains(module.content, "func.func @conv1d_gelu_add",
            message = "MLIR must contain the named function")

        // The conv output type must be tensor<1x8x16xf32>
        assertContains(module.content, "tensor<1x8x16xf32>",
            message = "Conv1d output shape must be [1, 8, 16] (batch=1, out_channels=8, " +
            "length=(16+2*1-3)/1+1=16)")

        // Bias is tensor<Cout> but conv output is tensor<N,Cout,L>. StableHLO requires an
        // explicit broadcast_in_dim before add — otherwise iree-compile rejects the MLIR with
        // "use of value '%vN' expects different type than prior uses".
        assertContains(module.content, "stablehlo.broadcast_in_dim",
            message = "Conv1d bias must be broadcast_in_dim'd before add to avoid iree-compile " +
            "type-mismatch on the bias operand")
    }

    /**
     * Verify conv1d with stride=2 computes the correct output length.
     *
     * input: [1, 3, 16], weight: [8, 3, 3], stride=2, padding=1
     * outLength = (16 + 2*1 - 3) / 2 + 1 = 8
     * expected output: [1, 8, 8]
     */
    @Test
    fun conv1d_stride2_output_shape_is_correct() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val input = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 3, 16), dtype = FP32::class,
            data = FloatArray(1 * 3 * 16)
        )
        val weight = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8, 3, 3), dtype = FP32::class,
            data = FloatArray(8 * 3 * 3)
        )
        val bias = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8), dtype = FP32::class,
            data = FloatArray(8)
        )

        @Suppress("UNCHECKED_CAST")
        val inputRefId = ctx.session.refOf(input as sk.ainet.lang.tensor.Tensor<*, *>).id

        val (tape, _) = ctx.record {
            ctx.ops.conv1d(input, weight, bias, stride = 2, padding = 1)
        }

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRefId)
        )

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "conv1d_stride2")

        println("MLIR:\n${module.content}")

        assertFalse(module.content.contains("tensor<?"),
            "Output must have static shape, not tensor<?xf32>")

        // outLength = (16 + 2*1 - 3) / 2 + 1 = 8
        assertContains(module.content, "tensor<1x8x8xf32>",
            message = "Conv1d stride=2 output shape must be [1, 8, 8]")
    }

    /**
     * Verify that matmul (linear layer pattern) has correct edge wiring:
     * transpose(weight) → matmul(input, weight_t) → add(bias)
     * Each op must have the correct number of inputs.
     */
    @Test
    fun matmul_add_edges_are_wired_correctly() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val input = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(4, 8), dtype = FP32::class,
            data = FloatArray(4 * 8)
        )
        val weight = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(16, 8), dtype = FP32::class,
            data = FloatArray(16 * 8)
        )
        val bias = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(16), dtype = FP32::class,
            data = FloatArray(16)
        )

        @Suppress("UNCHECKED_CAST")
        val inputRefId = ctx.session.refOf(input as sk.ainet.lang.tensor.Tensor<*, *>).id

        val (tape, _) = ctx.record {
            val wt = ctx.ops.transpose(weight)
            val mm = ctx.ops.matmul(input, wt)
            ctx.ops.add(mm, bias)
        }

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRefId)
        )

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "linear_layer")

        println("MLIR:\n${module.content}")

        // No unsupported ops — edges must be wired correctly
        assertFalse(module.content.contains("Unsupported"),
            "matmul and add must have correct input arity (2 each)")

        assertContains(module.content, "stablehlo.dot_general",
            message = "matmul should emit stablehlo.dot_general")

        assertContains(module.content, "stablehlo.add",
            message = "bias add should emit stablehlo.add")

        // bias is [16], matmul result is [4, 16] — rank-differing add must be broadcast
        // before stablehlo.add (iree-compile rejects mismatched ranks). This exercises
        // BasicMathConverter's rank-broadcast path (Issue C in whisper docs).
        assertContains(module.content, "stablehlo.broadcast_in_dim",
            message = "Rank-differing bias add must insert broadcast_in_dim")
    }

    /**
     * Issue B: traced graphs that use `sqrt`, `addScalar`, `mulScalar`, `subScalar`, or
     * `divScalar` must map to StableHLO. Before UnaryMathConverter/ScalarOperationsConverter
     * were added these ops fell through to "no converter found" and — because
     * StableHloConverter.processNode uses mapNotNull on operand lookups — cascade-broke the
     * arity of every downstream op (manifesting in the Whisper encoder as ~157 "Unsupported X
     * arity" comments over 296 nodes).
     */
    @Test
    fun scalar_and_unary_ops_produce_valid_stablehlo() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val x = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(2, 4), dtype = FP32::class,
            data = FloatArray(2 * 4)
        )

        @Suppress("UNCHECKED_CAST")
        val xRefId = ctx.session.refOf(x as sk.ainet.lang.tensor.Tensor<*, *>).id

        val (tape, _) = ctx.record {
            val a = ctx.ops.addScalar(x, 0.00001f)
            val b = ctx.ops.sqrt(a)
            val c = ctx.ops.mulScalar(b, 2.0f)
            val d = ctx.ops.subScalar(c, 0.5f)
            ctx.ops.divScalar(d, 4.0f)
        }

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(xRefId)
        )

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "scalar_sqrt")

        println("MLIR:\n${module.content}")

        assertFalse(module.content.contains("Unsupported"),
            "No ops should be unsupported once UnaryMath/ScalarOperations converters are registered")
        assertContains(module.content, "stablehlo.sqrt",
            message = "sqrt must lower to stablehlo.sqrt")
        // Scalar ops materialize a splat constant then apply the matching binary op.
        assertContains(module.content, "stablehlo.constant dense<",
            message = "scalar ops must materialize a splat constant")
        assertContains(module.content, "stablehlo.add",
            message = "addScalar must lower to stablehlo.add")
        assertContains(module.content, "stablehlo.multiply",
            message = "mulScalar must lower to stablehlo.multiply")
        assertContains(module.content, "stablehlo.subtract",
            message = "subScalar must lower to stablehlo.subtract")
        assertContains(module.content, "stablehlo.divide",
            message = "divScalar must lower to stablehlo.divide")
    }

    /**
     * Reproducer for the Whisper-encoder LayerNorm pattern: shared `x`, reduction
     * outputs consumed by subsequent ops, pre-recording scale/bias weights, and
     * a broadcast between `[N, D]` and `[N, 1]` (reduction result).
     *
     * The claim in whisper upstream-issues/TECHNICAL-ARTICLE-pipeline-analysis.md
     * is that `toComputeGraph()` drops edges for weight tensors / shared intermediates
     * so most layer-norm ops lose operands. This test attempts to reproduce that
     * on a minimal LayerNorm so we can see the actual failure mode in-tree.
     */
    @Test
    fun layer_norm_pattern_wires_all_edges() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        val x = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(2, 8), dtype = FP32::class,
            data = FloatArray(2 * 8)
        )
        // scale and bias are pre-recording weights, each shared by exactly one
        // consumer but with shape mismatched to the running tensor so BasicMath
        // must broadcast them.
        val scale = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8), dtype = FP32::class,
            data = FloatArray(8)
        )
        val bias = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(8), dtype = FP32::class,
            data = FloatArray(8)
        )

        @Suppress("UNCHECKED_CAST")
        val xRefId = ctx.session.refOf(x as sk.ainet.lang.tensor.Tensor<*, *>).id

        val (tape, _) = ctx.record {
            // mean: [2, 8] -> [2, 1] (keepdim-like via mean-then-unsqueeze is skipped —
            // VoidTensorOps.mean(dim=1) returns [2] per its reduction rule)
            val mean = ctx.ops.mean(x, dim = 1)              // [2]
            val variance = ctx.ops.variance(x, dim = 1)      // [2]
            val varEps = ctx.ops.addScalar(variance, 1e-5f)  // [2]
            val stddev = ctx.ops.sqrt(varEps)                // [2]
            // scale * x + bias  — skip full LN center/divide (would need unsqueeze-to-[2,1]
            // plus broadcast, which is a separate VoidTensorOps.divide shape check).
            // This test focuses on the edge-wiring question, so we use the simpler pattern:
            // multiply(x, scale_bcast) + bias_bcast. We explicitly broadcast scale/bias by
            // letting BasicMathConverter emit broadcast_in_dim (Issue C fix).
            val scaled = ctx.ops.multiply(x, scale)          // [2,8] * [8] — broadcast
            ctx.ops.add(scaled, bias)                        // [2,8] + [8] — broadcast
        }

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(xRefId)
        )

        val nodes = graph.getTopologicalOrder()
        println("Graph nodes (${nodes.size}):")
        for (n in nodes) {
            val inEdges = graph.getInputNodes(n).size
            println("  ${n.id} name=${n.operation.name} type=${n.operation.type} inputs.size=${n.inputs.size} wired=$inEdges")
        }

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "layer_norm_pattern")
        println("MLIR:\n${module.content}")

        assertFalse(module.content.contains("Unsupported"),
            "LayerNorm-style pattern must not have any Unsupported ops. Actual MLIR printed above.")
    }
}
