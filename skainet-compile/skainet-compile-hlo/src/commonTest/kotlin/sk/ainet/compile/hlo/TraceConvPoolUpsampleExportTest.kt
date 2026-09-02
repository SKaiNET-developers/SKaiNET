package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Regression for #1219: a TRACE-mode model (the `network{}` / `sequential{}` path,
 * i.e. `TensorOps` calls recorded through the KSP tracing wrapper — not the low-level
 * `dag{}` DSL) using `conv2d`, `maxPool2d` and `upsample2d` exported through
 * `toStableHlo` (which uses `StableHloConverterFactory.createBasic`) emitted an
 * `// Unsupported op` comment for every one of those nodes and still reported success.
 *
 * Two things closed it: `createBasic` now registers `NeuralNetOperationsConverter`
 * (#1248) — the "Known names" allowlist genuinely lacked convolution / pooling /
 * upsampling on the basic factory — and conversion is strict by default, so a
 * registry miss can no longer masquerade as a successful export. This test pins the
 * tracer's attribute spellings (`stride`/`padding`/`dilation`/`groups` as pairs,
 * `kernelSize`, `scale`/`mode`/`alignCorners`) against what the converter reads.
 */
class TraceConvPoolUpsampleExportTest {

    @Test
    fun trace_mode_conv2d_maxPool2d_upsample2d_lower_through_toStableHlo() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

        // NCHW input [1, 3, 8, 8]; conv weight [4, 3, 3, 3] (Cout, Cin, kH, kW); bias [4]
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 3, 8, 8), FP32::class, FloatArray(1 * 3 * 8 * 8))
        val weight = ctx.fromFloatArray<FP32, Float>(Shape(4, 3, 3, 3), FP32::class, FloatArray(4 * 3 * 3 * 3))
        val bias = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, FloatArray(4))

        @Suppress("UNCHECKED_CAST")
        val inputRefId = ctx.session.refOf(input as sk.ainet.lang.tensor.Tensor<*, *>).id

        val (tape, result) = ctx.record {
            // conv2d: same-padded 3x3 → [1, 4, 8, 8]
            val conv = ctx.ops.conv2d(input, weight, bias, stride = 1 to 1, padding = 1 to 1)
            // maxPool2d 2x2 / stride 2 → [1, 4, 4, 4]
            val pooled = ctx.ops.maxPool2d(conv, kernelSize = 2 to 2, stride = 2 to 2)
            // upsample2d x2 nearest → [1, 4, 8, 8]
            ctx.ops.upsample2d(pooled, scale = 2 to 2, mode = UpsampleMode.Nearest)
        }
        assertEquals(listOf(1, 4, 8, 8), result!!.shape.dimensions.toList(), "upsampled shape")

        val graph = tape!!.toComputeGraph(
            synthesizeExternalInputs = true,
            inputTensorIds = setOf(inputRefId)
        )
        val opNames = graph.getTopologicalOrder().map { it.operation.name }
        assertTrue("conv2d" in opNames && "maxPool2d" in opNames && "upsample2d" in opNames,
            "trace must record the three op names as the issue describes, got $opNames")

        // The issue's exact entry point (createBasic under the default STRICT policy):
        // under the pre-#1248 registry this threw on the first conv2d node (or, before
        // strictness, emitted `// Unsupported op 'conv2d'` and "succeeded").
        val module = toStableHlo(graph, functionName = "trace_conv_pool_upsample")
        val mlir = module.content

        assertFalse(mlir.contains("Unsupported op"), "no registry misses:\n$mlir")
        assertFalse(mlir.contains("Conversion failed"), "no converter failures:\n$mlir")
        assertFalse(mlir.contains("No output values"), "module must return a value:\n$mlir")

        assertContains(mlir, "stablehlo.convolution", message = "conv2d must lower to stablehlo.convolution:\n$mlir")
        assertContains(mlir, "reduce_window", message = "maxPool2d must lower to a reduce_window:\n$mlir")
        // Static shapes survive the trace path end to end.
        assertContains(mlir, "tensor<1x4x4x4xf32>", message = "pooled shape must be static:\n$mlir")
        assertContains(mlir, "tensor<1x4x8x8xf32>", message = "conv/upsample shape must be static:\n$mlir")
        assertFalse(mlir.contains("tensor<?"), "no dynamic shapes on a static trace:\n$mlir")
        assertContains(mlir, "func.func @trace_conv_pool_upsample", message = "named function:\n$mlir")
    }
}
