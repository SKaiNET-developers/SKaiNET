package sk.ainet.lang.tensor.ops

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TernaryTensorData
import sk.ainet.lang.types.FP32

/**
 * Optimized matrix multiplication for quantized weight formats (Q8_0, Q4_K).
 *
 * Direct quantized matmul avoids full dequantization to FP32, reducing memory
 * bandwidth and improving cache efficiency. The computation fuses dequantization
 * with the dot product:
 *
 * Q8_0: output[i] = sum(input[j] * code[j]) * scale
 * Q4_K: output[i] = sum(input[j] * code[j]) * scale + sum_of_mins
 *
 * These kernels maintain numerical accuracy within acceptable tolerance compared
 * to full dequant + FP32 matmul (typically ≤1e-4 relative error).
 */
public object QuantizedMatmul {

    /**
     * Matrix multiplication with Q8_0 quantized weights.
     *
     * @param input FP32 input tensor of shape [batch, inputDim] or [inputDim]
     * @param weights Q8_0 quantized weight data of shape [inputDim, outputDim]
     * @param ctx ExecutionContext for creating the output tensor
     * @return FP32 output tensor of shape [batch, outputDim] or [outputDim]
     */
    public fun matmulQ8_0(
        input: Tensor<FP32, Float>,
        weights: Q8_0TensorData,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        val inputData = input.data
        val inputShape = input.shape
        val weightsShape = weights.shape

        require(weightsShape.dimensions.size == 2) {
            "Q8_0 weights must be 2D, got shape ${weightsShape.dimensions.toList()}"
        }

        val inputDim = weightsShape.dimensions[0]
        val outputDim = weightsShape.dimensions[1]

        val isBatched = inputShape.dimensions.size == 2
        val batchSize = if (isBatched) inputShape.dimensions[0] else 1
        val inputLastDim = inputShape.dimensions[inputShape.dimensions.size - 1]

        require(inputLastDim == inputDim) {
            "Input last dimension ($inputLastDim) must match weight input dimension ($inputDim)"
        }

        val inputBuffer = getFloatBuffer(inputData)
        val outputShape = if (isBatched) Shape(batchSize, outputDim) else Shape(1, outputDim)
        val output = FloatArray(batchSize * outputDim)

        val blockSize = Q8_0TensorData.BLOCK_SIZE
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize

        for (b in 0 until batchSize) {
            val inputOffset = b * inputDim
            val outputOffset = b * outputDim

            for (o in 0 until outputDim) {
                var acc = 0f

                for (blockIdx in 0 until blocksPerInputDim) {
                    val weightBlockOffset = (blockIdx * outputDim + o)
                    val scale = weights.getBlockScale(weightBlockOffset)
                    var blockSum = 0f

                    val elemStart = blockIdx * blockSize
                    val elemEnd = minOf(elemStart + blockSize, inputDim)

                    for (i in elemStart until elemEnd) {
                        val code = weights.getCode(weightBlockOffset, i - elemStart)
                        blockSum += inputBuffer[inputOffset + i] * code.toFloat()
                    }

                    acc += blockSum * scale
                }

                output[outputOffset + o] = acc
            }
        }

        return ctx.fromFloatArray(outputShape, FP32::class, output)
    }

    /**
     * Matrix multiplication with Q4_K quantized weights.
     *
     * @param input FP32 input tensor of shape [batch, inputDim] or [inputDim]
     * @param weights Q4_K quantized weight data of shape [inputDim, outputDim]
     * @param ctx ExecutionContext for creating the output tensor
     * @return FP32 output tensor of shape [batch, outputDim] or [outputDim]
     */
    public fun matmulQ4_K(
        input: Tensor<FP32, Float>,
        weights: Q4_KTensorData,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        val inputData = input.data
        val inputShape = input.shape
        val weightsShape = weights.shape

        require(weightsShape.dimensions.size == 2) {
            "Q4_K weights must be 2D, got shape ${weightsShape.dimensions.toList()}"
        }

        val inputDim = weightsShape.dimensions[0]
        val outputDim = weightsShape.dimensions[1]

        val isBatched = inputShape.dimensions.size == 2
        val batchSize = if (isBatched) inputShape.dimensions[0] else 1
        val inputLastDim = inputShape.dimensions[inputShape.dimensions.size - 1]

        require(inputLastDim == inputDim) {
            "Input last dimension ($inputLastDim) must match weight input dimension ($inputDim)"
        }

        val inputBuffer = getFloatBuffer(inputData)
        val outputShape = if (isBatched) Shape(batchSize, outputDim) else Shape(1, outputDim)
        val output = FloatArray(batchSize * outputDim)

        val blockSize = Q4_KTensorData.BLOCK_SIZE
        val subBlockSize = Q4_KTensorData.SUB_BLOCK_SIZE
        val subBlocksPerBlock = Q4_KTensorData.SUB_BLOCKS_PER_BLOCK
        val blocksPerInputDim = (inputDim + blockSize - 1) / blockSize

        for (b in 0 until batchSize) {
            val inputOffset = b * inputDim
            val outputOffset = b * outputDim

            for (o in 0 until outputDim) {
                var acc = 0f

                for (blockIdx in 0 until blocksPerInputDim) {
                    val weightBlockOffset = blockIdx * outputDim + o

                    for (subBlockIdx in 0 until subBlocksPerBlock) {
                        val scale = weights.getSubBlockScale(weightBlockOffset, subBlockIdx)
                        val offset = weights.getSubBlockMin(weightBlockOffset, subBlockIdx)

                        val elemStart = blockIdx * blockSize + subBlockIdx * subBlockSize
                        val elemEnd = minOf(elemStart + subBlockSize, inputDim)

                        if (elemStart >= inputDim) break

                        var subBlockSum = 0f
                        var inputSum = 0f

                        for (i in elemStart until elemEnd) {
                            val localIdx = i - blockIdx * blockSize
                            val code = weights.getCode(weightBlockOffset, localIdx)
                            subBlockSum += inputBuffer[inputOffset + i] * code.toFloat()
                            inputSum += inputBuffer[inputOffset + i]
                        }

                        // ggml's per-element formula `code * scale - offset` aggregates
                        // to `subBlockSum * scale - inputSum * offset` over the sub-block.
                        acc += subBlockSum * scale - inputSum * offset
                    }
                }

                output[outputOffset + o] = acc
            }
        }

        return ctx.fromFloatArray(outputShape, FP32::class, output)
    }

    /**
     * Check if a tensor's underlying data is Q8_0 quantized.
     */
    public fun isQ8_0Weight(tensor: Tensor<*, *>): Boolean {
        return tensor.data is Q8_0TensorData
    }

    /**
     * Check if a tensor's underlying data is Q4_K quantized.
     */
    public fun isQ4_KWeight(tensor: Tensor<*, *>): Boolean {
        return tensor.data is Q4_KTensorData
    }

    /**
     * Check if a tensor's underlying data is any quantized format we support.
     */
    public fun isQuantizedWeight(tensor: Tensor<*, *>): Boolean {
        val data = tensor.data
        return data is Q8_0TensorData || data is Q4_KTensorData || data is TernaryTensorData
    }

    /**
     * Perform matmul with automatic dispatch based on weight type.
     * Uses quantized-optimized path when weights are quantized,
     * otherwise falls back to standard matmul.
     *
     * Supported weight types:
     * - Q8_0TensorData: Uses Q8_0 fused matmul
     * - Q4_KTensorData: Uses Q4_K fused matmul
     * - TernaryTensorData: Uses ternary addition-only matmul
     * - FP32: Standard floating-point matmul
     *
     * @param input FP32 input tensor
     * @param weight Weight tensor (quantized or FP32)
     * @param ctx ExecutionContext
     * @return Output tensor
     */
    public fun matmulAutoDispatch(
        input: Tensor<FP32, Float>,
        weight: Tensor<*, *>,
        ctx: ExecutionContext
    ): Tensor<FP32, Float> {
        val weightData = weight.data
        return when (weightData) {
            is Q8_0TensorData -> matmulQ8_0(input, weightData, ctx)
            is Q4_KTensorData -> matmulQ4_K(input, weightData, ctx)
            is TernaryTensorData -> TernaryMatmul.matmul(input, weightData, ctx)
            else -> {
                @Suppress("UNCHECKED_CAST")
                ctx.ops.matmul(input, weight as Tensor<FP32, Float>)
            }
        }
    }

    private fun getFloatBuffer(data: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatArray {
        return when (data) {
            is FloatArrayTensorData<*> -> data.buffer
            else -> {
                val shape = data.shape
                val volume = shape.volume
                val dims = shape.dimensions
                FloatArray(volume) { flatIdx ->
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
 * Extension function for Q8_0 matmul.
 */
public fun Tensor<FP32, Float>.matmulQ8_0(
    weights: Q8_0TensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = QuantizedMatmul.matmulQ8_0(this, weights, ctx)

/**
 * Extension function for Q4_K matmul.
 */
public fun Tensor<FP32, Float>.matmulQ4_K(
    weights: Q4_KTensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = QuantizedMatmul.matmulQ4_K(this, weights, ctx)
