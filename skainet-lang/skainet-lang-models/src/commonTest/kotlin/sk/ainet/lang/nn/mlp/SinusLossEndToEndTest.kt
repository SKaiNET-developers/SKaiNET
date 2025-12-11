package sk.ainet.lang.nn.mlp

import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.loss.evaluateLoss
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.FP32

class SinusLossEndToEndTest {
    private val ctx = DirectCpuExecutionContext()
    private val loss = MSELoss()

    @Test
    fun pretrained_beats_zero_baseline_on_sine_batch() {
        val inputs = floatArrayOf(-PI.toFloat() / 2f, 0f, PI.toFloat() / 2f)
        val targets = inputs.map { sin(it) }.toFloatArray()

        val xTensor = data<FP32, Float>(ctx) {
            tensor {
                shape(3, 1) {
                    fromArray(inputs)
                }
            }
        }
        val yTensor = data<FP32, Float>(ctx) {
            tensor {
                shape(3, 1) {
                    fromArray(targets)
                }
            }
        }

        val pretrained = SinusApproximator().create(ctx)
        // Use the same architecture without pretrained weights as the baseline (instead of ZeroModule)
        val untrainedBaseline = createEmptySinusApproximator(ctx)

        // Pipeline-style: model forward combined with explicit loss evaluation helper
        val pretrainedLoss = evaluateLoss(pretrained, loss, xTensor, yTensor, ctx).data.get() as Float
        val zeroLoss = evaluateLoss(untrainedBaseline, loss, xTensor, yTensor, ctx).data.get() as Float

        assertTrue(pretrainedLoss < zeroLoss, "Pretrained sinus approximator should beat zero baseline")
    }

    private fun createEmptySinusApproximator(executionContext: DirectCpuExecutionContext): Module<FP32, Float> =
        definition {
            network(executionContext) {
                input(1, "input")  // Single input for x value

                // First hidden layer: 1 -> 16 neurons
                dense(16, "hidden-1") {
                    // Weights: 16x1 matrix - explicitly defined values
                    weights {
                        zeros()
                    }
                    // Bias: 16 values - explicitly defined
                    bias {
                        zeros()
                    }
                    activation = { tensor -> with(tensor) { relu() } }
                }

                // Second hidden layer: 16 -> 16 neurons
                dense(16, "hidden-2") {
                    // Weights: 16x16 matrix - explicitly defined values
                    weights {
                        zeros()
                    }
                    // Bias: 16 values - explicitly defined
                    bias {
                        zeros()
                    }
                    activation = { tensor -> with(tensor) { relu() } }
                }

                // Output layer: 16 -> 1 neuron
                dense(1, "output") {
                    // Weights: 1x16 matrix - explicitly defined values
                    weights {
                        zeros()
                    }

                    // Bias: single value - explicitly defined
                    bias {
                        zeros()
                    }
                    // No activation for output layer (linear output)
                }
            }
        }
}
