package sk.ainet.lang.model.dnn.cnn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.ModelCard
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.definition
import sk.ainet.lang.nn.dsl.sequential
import sk.ainet.lang.nn.network
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.types.FP32

/**
 * Constructs a Convolutional Neural Network (CNN) tailored for the MNIST dataset using a DSL-based network builder.
 *
 * This model consists of two convolutional blocks followed by a flattening stage and two dense (fully connected) layers.
 * It is designed to classify handwritten digits (0–9) from grayscale 28x28 pixel images.
 *
 * The architecture is as follows:
 *
 * - Stage: "conv1"
 *   - 2D Convolution with:
 *     - 16 output channels
 *     - 5x5 kernel
 *     - stride of 1
 *     - padding of 2
 *   - ReLU activation
 *   - 2x2 MaxPooling with stride of 2
 *
 * - Stage: "conv2"
 *   - 2D Convolution with:
 *     - 32 output channels
 *     - 5x5 kernel
 *     - stride of 1
 *     - padding of 2
 *   - ReLU activation
 *   - 2x2 MaxPooling with stride of 2
 *
 * - Stage: "flatten"
 *   - Flattens the tensor for dense layer input
 *
 * - Stage: "dense"
 *   - Fully connected layer with 128 units
 *   - ReLU activation
 *
 * - Stage: "output"
 *   - Fully connected layer with 10 output units (for 10 MNIST classes)
 *   - Softmax activation over dimension 1 to produce class probabilities
 */
public class MnistCnn : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    private fun buildModel(executionContext: ExecutionContext): Module<FP32, Float> = definition {
        network(executionContext) {
            sequential<FP32, Float> {
                // Per-sample input shape (channels, height, width). Required for the
                // DSL to track shapes through conv/pool stages and infer the flatten
                // size before the dense layers.
                input(intArrayOf(1, 28, 28))

                // Stage: "conv1"
                stage("conv1") {
                    conv2d(outChannels = 16, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu1") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }

                // Stage: "conv2"
                stage("conv2") {
                    conv2d(outChannels = 32, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
                    activation(id = "relu2") { tensor -> tensor.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }

                // Stage: "flatten"
                stage("flatten") {
                    flatten()
                }

                // Stage: "dense"
                stage("dense") {
                    dense(outputDimension = 128)
                    activation(id = "relu3") { tensor -> tensor.relu() }
                }

                // Stage: "output"
                stage("output") {
                    dense(outputDimension = 10)
                    activation(id = "softmax") { tensor -> tensor.softmax(dim = 1) }
                }
            }
        }
    }

    // Backward-compatible helper for old call sites
    public fun model(executionContext: ExecutionContext): Module<FP32, Float> = create(executionContext)

    override fun create(executionContext: ExecutionContext): Module<FP32, Float> = buildModel(executionContext)

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        executionContext: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): Tensor<FP32, Float> {
        reportProgress(0, 1, "starting mnist-cnn forward")
        val out = module.forward(inputValue, executionContext)
        reportProgress(1, 1, "done")
        return out
    }

    override fun modelCard(): ModelCard {
        return ModelCard(
            license = "apache-2.0",
            libraryName = "skainet",
            pipelineTag = "image-classification",
            language = listOf("en"),
            modalities = listOf("vision"),
            baseModel = "",
            contextLength = 0,
            datasets = listOf("mnist"),
            metrics = emptyList(),
            modelIndex = emptyList(),
            intendedUse = "MNIST CNN: 2x Conv(16,5x5)->MaxPool -> 2x Conv(32,5x5)->MaxPool -> Flatten -> Dense(128, ReLU) -> Dense(10, Softmax)",
            limitations = ""
        )
    }
}
