package sk.ainet.lang.tensor.ops

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TernaryTensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.types.FP32

/**
 * Optimized matrix multiplication for BitNet-style ternary weights.
 *
 * When the weight matrix contains only {-1, 0, +1} values, multiplication
 * becomes addition-only, which is significantly faster:
 *
 * ```
 * output[i] = sum over j of: activation[j] * ternary_weight[j,i]
 *           = sum where weight=+1: activation[j]
 *           - sum where weight=-1: activation[j]
 * ```
 *
 * This avoids all floating-point multiplications, replacing them with
 * conditional additions/subtractions based on the ternary weight value.
 */
public object TernaryMatmul {

    /**
     * Perform matrix multiplication with ternary weights.
     *
     * @param input FP32 input tensor of shape [batch, inputDim] or [inputDim]
     * @param ternaryWeights Ternary weight tensor of shape [inputDim, outputDim]
     * @param ctx ExecutionContext for creating the output tensor
     * @return FP32 output tensor of shape [batch, outputDim] or [outputDim]
     */
    public fun matmul(
        input: Tensor<FP32, Float>,
        ternaryWeights: TernaryTensorData,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        val inputData = input.data
        val inputShape = input.shape
        val weightsShape = ternaryWeights.shape

        require(weightsShape.dimensions.size == 2) {
            "Ternary weights must be 2D, got shape ${weightsShape.dimensions.toList()}"
        }

        val inputDim = weightsShape.dimensions[0]
        val outputDim = weightsShape.dimensions[1]

        // Handle both 1D and 2D input
        val isBatched = inputShape.dimensions.size == 2
        val batchSize = if (isBatched) inputShape.dimensions[0] else 1
        val inputLastDim = inputShape.dimensions[inputShape.dimensions.size - 1]

        require(inputLastDim == inputDim) {
            "Input last dimension ($inputLastDim) must match weight input dimension ($inputDim)"
        }

        // Get input as FloatArray for efficient access
        val inputBuffer = getFloatBuffer(inputData)

        // Perform ternary matmul
        val outputShape = if (isBatched) Shape(batchSize, outputDim) else Shape(1, outputDim)
        val output = FloatArray(batchSize * outputDim)

        // Apply scale from ternary weights (typically 1.0 for properly quantized models)
        val scale = ternaryWeights.scale

        for (b in 0 until batchSize) {
            val inputOffset = b * inputDim
            val outputOffset = b * outputDim

            for (o in 0 until outputDim) {
                var sum = 0f

                // Core ternary matmul: addition-only
                for (i in 0 until inputDim) {
                    val ternaryVal = ternaryWeights[i, o].toInt()
                    when (ternaryVal) {
                        1 -> sum += inputBuffer[inputOffset + i]
                        -1 -> sum -= inputBuffer[inputOffset + i]
                        // 0 -> do nothing
                    }
                }

                output[outputOffset + o] = sum * scale
            }
        }

        return ctx.fromFloatArray(outputShape, FP32::class, output)
    }

    /**
     * Check if a tensor's underlying data is ternary.
     * This can be used to dispatch to optimized ternary matmul.
     */
    public fun isTernaryWeight(tensor: Tensor<*, *>): Boolean {
        return tensor.data is TernaryTensorData
    }

    /**
     * Perform matmul with automatic dispatch based on weight type.
     * Uses ternary-optimized path when weights are TernaryTensorData,
     * otherwise falls back to standard matmul.
     *
     * @param input FP32 input tensor
     * @param weight Weight tensor (either FP32 or Ternary)
     * @param ctx ExecutionContext
     * @return Output tensor
     */
    public fun matmulAutoDispatch(
        input: Tensor<FP32, Float>,
        weight: Tensor<*, *>,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        val weightData = weight.data
        return if (weightData is TernaryTensorData) {
            matmul(input, weightData, ctx)
        } else {
            // Fall back to standard matmul
            @Suppress("UNCHECKED_CAST")
            ctx.ops.matmul(input, weight as Tensor<FP32, Float>)
        }
    }

    private fun getFloatBuffer(data: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatArray {
        return when (data) {
            is FloatArrayTensorData<*> -> data.buffer
            else -> {
                // Generic fallback - iterate through all elements
                val shape = data.shape
                val volume = shape.volume
                val dims = shape.dimensions
                FloatArray(volume) { flatIdx ->
                    // Convert flat index to multi-dimensional indices
                    val indices = IntArray(dims.size)
                    var remaining = flatIdx
                    for (d in dims.size - 1 downTo 0) {
                        indices[d] = remaining % dims[d]
                        remaining /= dims[d]
                    }
                    @Suppress("UNCHECKED_CAST")
                    val value = (data as sk.ainet.lang.tensor.data.TensorData<*, Float>).get(*indices)
                    value
                }
            }
        }
    }
}

/**
 * Extension function for convenient ternary matmul.
 * Use this when you know the weight is ternary-quantized.
 */
public fun Tensor<FP32, Float>.matmulTernary(
    weights: TernaryTensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = TernaryMatmul.matmul(this, weights, ctx)
