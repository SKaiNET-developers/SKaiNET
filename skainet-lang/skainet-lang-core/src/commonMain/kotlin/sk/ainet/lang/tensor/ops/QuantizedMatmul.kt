package sk.ainet.lang.tensor.ops

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
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
 *
 * Deprecated (SKEEP-003, #1507): `ops.matmul` chooses the kernel for a packed weight from the
 * weight's view, for every encoding and storage kind. These helpers are scalar, know two formats
 * and are typed on the data interfaces.
 */
@Deprecated(
    "ops.matmul(input, weight) chooses the kernel for a packed weight from its view; these scalar helpers know two formats",
    ReplaceWith("ctx.ops.matmul(input, weight)"),
)
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

    /** The encoding the tensor's view reports, `null` for data without a view. */
    private fun encodingOf(tensor: Tensor<*, *>): TensorEncoding? = tensor.data.view?.format?.encoding

    /**
     * Whether the tensor's view reports the Q8_0 encoding.
     */
    public fun isQ8_0Weight(tensor: Tensor<*, *>): Boolean = encodingOf(tensor) == TensorEncoding.Q8_0

    /**
     * Whether the tensor's view reports the Q4_K encoding.
     */
    public fun isQ4_KWeight(tensor: Tensor<*, *>): Boolean = encodingOf(tensor) == TensorEncoding.Q4_K

    /**
     * Whether the tensor's view reports one of the encodings these helpers know: Q8_0, Q4_K or
     * packed ternary.
     */
    public fun isQuantizedWeight(tensor: Tensor<*, *>): Boolean = when (encodingOf(tensor)) {
        TensorEncoding.Q8_0, TensorEncoding.Q4_K, TensorEncoding.TernaryPacked -> true
        else -> false
    }

    /**
     * `ctx.ops.matmul(input, weight)`. The ops choose the kernel from the weight's view, so a
     * packed weight of any encoding takes its packed kernel and a dense one the dense kernel.
     *
     * @param input FP32 input tensor
     * @param weight Weight tensor (packed or FP32), `[inputDim, outputDim]`
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
 * Extension function for Q8_0 matmul.
 */
@Suppress("DEPRECATION")
@Deprecated("ops.matmul(input, weight) chooses the kernel for a packed weight from its view")
public fun Tensor<FP32, Float>.matmulQ8_0(
    weights: Q8_0TensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = QuantizedMatmul.matmulQ8_0(this, weights, ctx)

/**
 * Extension function for Q4_K matmul.
 */
@Suppress("DEPRECATION")
@Deprecated("ops.matmul(input, weight) chooses the kernel for a packed weight from its view")
public fun Tensor<FP32, Float>.matmulQ4_K(
    weights: Q4_KTensorData,
    ctx: ExecutionContext
): Tensor<FP32, Float> = QuantizedMatmul.matmulQ4_K(this, weights, ctx)
