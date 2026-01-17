package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * 2D Average Pooling layer that applies an average pooling operation over 2D input.
 *
 * Average pooling reduces the spatial dimensions of the input by computing the mean
 * value in each pooling window. This is often used as an alternative to max pooling
 * and can help preserve more information about the input.
 *
 * @param kernelSize Size of the pooling window (height, width)
 * @param stride Stride of the pooling operation (default: same as kernelSize)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param countIncludePad Whether to include padding in the average calculation (default: true)
 * @param name Name of the module
 */
public class AvgPool2d<T : DType, V>(
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = kernelSize,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val countIncludePad: Boolean = true,
    override val name: String = "AvgPool2d",
) : Module<T, V>() {

    init {
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            input.ops.avgPool2d(
                input = input,
                kernelSize = kernelSize,
                stride = stride,
                padding = padding,
                countIncludePad = countIncludePad
            )
        }

    /**
     * Calculates the output size for a given input size and pooling parameters.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        val (inputHeight, inputWidth) = inputSize
        val (kernelHeight, kernelWidth) = kernelSize
        val (strideHeight, strideWidth) = stride
        val (padHeight, padWidth) = padding

        val outputHeight = ((inputHeight + 2 * padHeight - kernelHeight) / strideHeight) + 1
        val outputWidth = ((inputWidth + 2 * padWidth - kernelWidth) / strideWidth) + 1

        return outputHeight to outputWidth
    }
}
