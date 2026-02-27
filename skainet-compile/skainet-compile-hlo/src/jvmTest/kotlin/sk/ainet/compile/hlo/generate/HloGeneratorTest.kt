package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.test.runTest
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul
import sk.ainet.lang.tensor.Shape
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
        // Conv2D ops are traced even if the basic converter emits them as comments
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
        assertTrue(module.content.length > 50, "Expected non-trivial MLIR output")
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
