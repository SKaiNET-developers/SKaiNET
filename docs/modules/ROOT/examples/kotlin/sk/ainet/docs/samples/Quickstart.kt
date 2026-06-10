package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.FP32
// end::imports[]

/**
 * Flagship "hello world" for the SKaiNET model DSL: define a small MLP with the
 * `sequential { }` DSL, then run a single forward pass on the CPU.
 *
 * Every region below is included verbatim into
 * `tutorials/kotlin-getting-started.adoc`. The companion test
 * (`QuickstartTest`) runs this end-to-end so the snippet can never drift.
 */
object Quickstart {

    /**
     * Build a `784 -> 128 (ReLU) -> 10 (Softmax)` classifier purely in code.
     * This is the SKaiNET "spirit": the network *is* a Kotlin DSL block.
     */
    // tag::model[]
    fun buildModel(ctx: DirectCpuExecutionContext) =
        sequential<FP32, Float>(ctx) {
            input(784)                                  // 28x28 flattened
            dense(128) { activation = { it.relu() } }   // hidden layer
            dense(10) { activation = { it.softmax(1) } } // class scores
        }
    // end::model[]

    /**
     * Run one forward pass over a batch of one sample and return the 10 class scores.
     */
    // tag::infer[]
    fun classify(pixels: FloatArray): Tensor<FP32, Float> {
        val ctx = DirectCpuExecutionContext.create()
        val model = buildModel(ctx)

        // Shape is [batch, features]; one sample here.
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, 784), FP32::class, pixels)

        return model.forward(input, ctx)               // [1, 10] class scores
    }
    // end::infer[]
}
