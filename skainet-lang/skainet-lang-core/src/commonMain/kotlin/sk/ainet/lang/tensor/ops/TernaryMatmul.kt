package sk.ainet.lang.tensor.ops

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TernaryTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
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
 *
 * Deprecated (SKEEP-003, #1507): `ops.matmul` reads a ternary weight through its view. This helper
 * reads the weight one element at a time and is typed on the data interface.
 */
@Deprecated(
    "ops.matmul(input, weight) reads a ternary weight through its view; this helper is scalar and typed on the data interface",
    ReplaceWith("ctx.ops.matmul(input, weight)"),
)
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
     * Whether the tensor's view reports the packed ternary encoding.
     */
    public fun isTernaryWeight(tensor: Tensor<*, *>): Boolean =
        tensor.data.view?.format?.encoding == TensorEncoding.TernaryPacked

    /**
     * `ctx.ops.matmul(input, weight)`. The ops read a ternary weight through its view, scale
     * included, and a dense weight as before.
     *
     * @param input FP32 input tensor
     * @param weight Weight tensor (either FP32 or Ternary), `[inputDim, outputDim]`
     * @param ctx ExecutionContext
     * @return Output tensor
     */
    public fun matmulAutoDispatch(
        input: Tensor<FP32, Float>,
        weight: Tensor<*, *>,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        @Suppress("UNCHECKED_CAST")
        return ctx.ops.matmul(input, weight as Tensor<FP32, Float>)
    }

    /** The values in row-major order: the data's own array when it is one from element 0, else a copy. */
    private fun getFloatBuffer(data: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatArray {
        val floats = data.heapFloats
        return if (floats != null && data.heapFloatsOffset == 0 && floats.size == data.shape.volume) floats
        else data.copyToFloatArray()
    }
}

/**
 * Extension function for convenient ternary matmul.
 * Use this when you know the weight is ternary-quantized.
 */
@Suppress("DEPRECATION")
@Deprecated("ops.matmul(input, weight) reads a ternary weight through its view")
public fun Tensor<FP32, Float>.matmulTernary(
    weights: TernaryTensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = TernaryMatmul.matmul(this, weights, ctx)
