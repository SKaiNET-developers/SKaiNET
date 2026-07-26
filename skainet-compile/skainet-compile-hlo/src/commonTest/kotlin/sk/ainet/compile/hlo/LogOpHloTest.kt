package sk.ainet.compile.hlo

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.log
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import sk.ainet.tape.Execution
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Coverage for the `log` (natural logarithm) tensor op — added so learned frontends (e.g. Moonshine v2's
 * asinh compression, `asinh(x) = log(x + sqrt(x²+1))`) can be authored in the DSL. Confirms the extension
 * `Tensor.log()` traces through to a `stablehlo.log` primitive (the [UnaryMathConverter] mapping).
 */
class LogOpHloTest {
    @Test
    fun logTracesToStableHloLog() {
        val input = VoidOpsTensor(
            object : TensorData<FP32, Float> {
                override val shape = Shape(2, 3)
                override fun get(vararg indices: Int): Float = 1.0f
                override fun set(vararg indices: Int, value: Float) {}
            },
            FP32::class,
        )
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        val tape = ctx.record {
            val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
            Execution.tapeStack.pushTape(ct)
            try {
                input.bind(this as ExecutionContext).log()   // bind to the recording ctx so the op is traced
            } finally {
                Execution.tapeStack.popTape()
            }
        }.first
        val graph = (tape as DefaultExecutionTape).toComputeGraph(synthesizeExternalInputs = true)
        val mlir = sk.ainet.compile.hlo.toStableHlo(graph, "log_test").content
        assertTrue(mlir.contains("stablehlo.log"), "Tensor.log() must lower to stablehlo.log; got:\n$mlir")
    }
}
