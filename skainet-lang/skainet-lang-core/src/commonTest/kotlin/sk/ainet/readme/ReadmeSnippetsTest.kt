package sk.ainet.readme

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32

class ReadmeSnippetsTest {

    @Test
    fun `tensor creation and indexing example`() {
        val ctx: ExecutionContext = DefaultNeuralNetworkExecutionContext()
        val t = data<FP32, Float>(ctx) {
            tensor{
                shape(2, 3) {
                    from(
                        0f, 1f, 2f,
                        10f, 11f, 12f
                    )
                }
            }
        }
        assertEquals(Shape(2, 3), t.shape)
        assertEquals(0f, t.data[0, 0])
        assertEquals(11f, t.data[1, 1])
    }

    @Test
    fun `tiny NN forward in eval and train contexts`() {
        val ctxEval = DefaultNeuralNetworkExecutionContext()
        val model = sequential<FP32, Float> {
            input(28 * 28)
            dense(128)
            activation { tensor -> tensor.relu() }
            dense(10)
        }

        val x = tensor<FP32, Float>(ctxEval, FP32::class) {
            tensor {
                shape(1, 28 * 28) {
                    full(0.5f)
                }
            }
        }

        val yEval = model.forward(x, ctxEval)
        assertEquals(2, yEval.shape.rank)
        assertEquals(1, yEval.shape[0])
        assertEquals(10, yEval.shape[1])

        val ctxTrain = DefaultNeuralNetworkExecutionContext(phase = sk.ainet.context.Phase.TRAIN)
        val yTrain = model.forward(x, ctxTrain)
        // Basic sanity checks
        assertEquals(yEval.shape, yTrain.shape)
        assertTrue(yTrain.volume == 10)
    }
}
