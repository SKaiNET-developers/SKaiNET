package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.test.runTest
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.lang.dag.dag
import sk.ainet.lang.dag.multiply
import sk.ainet.lang.dag.sum
import sk.ainet.lang.dag.unsqueeze
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertTrue

class HloGeneratorTest {

    @Test
    fun testGenerateWithRgb2GrayScale() = runTest {
        val model = Rgb2GrayScale()
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val sampleInput = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 3, 4, 4),
            dtype = FP32::class,
            data = FloatArray(1 * 3 * 4 * 4) { 0.5f }
        )

        val module = HloGenerator.generate(model, sampleInput, "rgb2grayscale")

        assertTrue(module.content.contains("module {"), "Expected 'module {' in MLIR output")
        assertTrue(module.content.contains("func.func"), "Expected 'func.func' in MLIR output")
        assertTrue(module.content.contains("@rgb2grayscale"), "Expected function name in MLIR output")
        // Extended converter now handles conv2d, but VoidTensorOps tracing may not yet
        // produce operations in the compute graph. Assert structure only for now.
        assertTrue(module.content.length > 50, "Expected non-trivial MLIR output")
    }

    @Test
    fun testGenerateWithRgb2GrayScaleMatMul() = runTest {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val model = Rgb2GrayScaleMatMul(ctx)
        val sampleInput = ctx.fromFloatArray<FP16, Float>(
            shape = Shape(1, 3, 4, 4),
            dtype = FP16::class,
            data = FloatArray(1 * 3 * 4 * 4) { 0.5f }
        )

        val module = HloGenerator.generate(model, sampleInput, "rgb2grayscale_matmul")

        assertTrue(module.content.contains("module {"), "Expected 'module {' in MLIR output")
        assertTrue(module.content.contains("func.func"), "Expected 'func.func' in MLIR output")
        assertTrue(module.content.contains("@rgb2grayscale_matmul"), "Expected function name in MLIR output")
        assertTrue(module.content.contains("stablehlo."), "Expected tensor-bound forward pass to emit StableHLO ops")
        assertTrue(module.content.length > 50, "Expected non-trivial MLIR output")
    }

    @Test
    fun testDagConstantsAreInlinedAndReductionDropsDimension() {
        val h = 8
        val w = 8
        val program = dag {
            val x = input<FP16>("input", TensorSpec("input", listOf(1, 3, h, w), "FP16"))
            val luma = constant<FP16, Float>("luma") {
                fromArray(floatArrayOf(0.2989f, 0.5870f, 0.1140f), shape = listOf(1, 3, 1, 1))
            }
            val weighted = multiply(x, luma)
            val grayHW = sum(weighted, 1)
            output(unsqueeze(grayHW, 1))
        }

        val mlir = StableHloConverterFactory.createExtended()
            .convert(program.toComputeGraph(), "grayscale")
            .content

        assertTrue(
            mlir.contains("stablehlo.constant") && !Regex("""@grayscale\([^)]*,[^)]*\)""").containsMatchIn(mlir),
            "luma constant should be baked into the StableHLO module:\n$mlir"
        )
        assertTrue(
            mlir.contains("tensor<1x1x${h}x${w}xf16>"),
            "sum over the channel dimension should drop that dimension before unsqueeze:\n$mlir"
        )
    }

    @Test
    fun testGenerateDefaultFunctionName() = runTest {
        val model = Rgb2GrayScale()
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val sampleInput = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 3, 4, 4),
            dtype = FP32::class,
            data = FloatArray(1 * 3 * 4 * 4) { 0.5f }
        )

        val module = HloGenerator.generate(model, sampleInput)

        assertTrue(module.content.contains("@main"), "Expected default function name '@main' in MLIR output")
    }

    @Test
    fun testGenerateCustomFunctionName() = runTest {
        val model = Rgb2GrayScale()
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val sampleInput = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(1, 3, 4, 4),
            dtype = FP32::class,
            data = FloatArray(1 * 3 * 4 * 4) { 0.5f }
        )

        val module = HloGenerator.generate(model, sampleInput, "custom_fn")

        assertTrue(module.content.contains("@custom_fn"), "Expected custom function name '@custom_fn' in MLIR output")
    }

    @Test
    fun testGenerateFromDescriptor() = runTest {
        val descriptor = ModelRegistry.get("rgb2grayscale")
            ?: error("Expected 'rgb2grayscale' in ModelRegistry")

        val module = HloGenerator.generate(descriptor, height = 4, width = 4, batch = 1)

        assertTrue(module.content.contains("module {"), "Expected 'module {' in MLIR output")
        assertTrue(module.content.contains("func.func"), "Expected 'func.func' in MLIR output")
        assertTrue(module.content.contains("@rgb2grayscale"), "Expected function name '@rgb2grayscale' in MLIR output")
    }
}
