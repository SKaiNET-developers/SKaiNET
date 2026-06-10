package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.test.runTest
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertTrue

class HloGeneratorCommonTest {

    @Test
    fun commonGeneratorEmitsStableHloOps() = runTest {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val sampleInput = ctx.fromFloatArray<FP32, Float>(
            shape = Shape(2, 2),
            dtype = FP32::class,
            data = floatArrayOf(1f, 2f, 3f, 4f)
        )

        val module = HloGenerator.generate(SquareModel(), sampleInput, "square")

        assertTrue(module.content.contains("func.func @square"), "Expected generated function name")
        assertTrue(module.content.contains("stablehlo.multiply"), "Expected traced multiply op")
        assertTrue(module.content.contains("tensor<2x2xf32>"), "Expected input/output tensor type")
    }

    private class SquareModel : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {
        override fun create(executionContext: ExecutionContext): Module<FP32, Float> = object : Module<FP32, Float>() {
            override val name: String = "square"
            override val modules: List<Module<FP32, Float>> = emptyList()
        }

        override suspend fun calculate(
            module: Module<FP32, Float>,
            inputValue: Tensor<FP32, Float>,
            executionContext: ExecutionContext,
            reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
        ): Tensor<FP32, Float> = inputValue * inputValue

        override fun modelCard(): ModelCard = error("Not needed for HLO generator tests")
    }
}
