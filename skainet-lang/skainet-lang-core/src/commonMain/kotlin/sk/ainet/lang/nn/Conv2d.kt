package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.topology.bias
import sk.ainet.lang.nn.topology.weights

/**
 * 2D Convolutional layer that applies a convolution operation over 2D input.
 * 
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels/filters
 * @param kernelSize Size of the convolving kernel (height, width)
 * @param stride Stride of the convolution (default: 1, 1)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param dilation Spacing between kernel elements (default: 1, 1)
 * @param groups Number of blocked connections from input channels to output channels (default: 1)
 * @param bias Whether to add a learnable bias to the output (default: true)
 * @param name Name of the module
 * @param initWeights Initial weights tensor
 * @param initBias Initial bias tensor (if bias is true)
 */
public class Conv2d<T : DType, V> @kotlin.jvm.JvmOverloads constructor(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = 1 to 1,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val dilation: Pair<Int, Int> = 1 to 1,
    public val groups: Int = 1,
    public val bias: Boolean = true,
    override val name: String = "Conv2d",
    initWeights: Tensor<T, V>,
    initBias: Tensor<T, V>? = null,
    public val trainable: Boolean = true
) : Module<T, V>(), ModuleParameters<T, V> {
    
    init {
        require(inChannels > 0) { "inChannels must be positive" }
        require(outChannels > 0) { "outChannels must be positive" }
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
        require(dilation.first > 0 && dilation.second > 0) { "dilation must be positive" }
        require(groups > 0) { "groups must be positive" }
        require(inChannels % groups == 0) { "inChannels must be divisible by groups" }
        require(outChannels % groups == 0) { "outChannels must be divisible by groups" }
    }

    override val params: List<ModuleParameter<T, V>> = buildList {
        add(ModuleParameter.WeightParameter("$name.weight", initWeights, trainable))
        if (bias && initBias != null) {
            add(ModuleParameter.BiasParameter("$name.bias", initBias, trainable))
        }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val weight = params.weights().value
            val biasValue = if (bias) params.bias().value else null

            input.ops.conv2d(
                input = input,
                weight = weight,
                bias = biasValue,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = groups
            )
        }

    /**
     * Calculates the output size for a given input size and convolution parameters.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        val (inputHeight, inputWidth) = inputSize
        val (kernelHeight, kernelWidth) = kernelSize
        val (strideHeight, strideWidth) = stride
        val (padHeight, padWidth) = padding
        val (dilationHeight, dilationWidth) = dilation
        
        val outputHeight = ((inputHeight + 2 * padHeight - dilationHeight * (kernelHeight - 1) - 1) / strideHeight) + 1
        val outputWidth = ((inputWidth + 2 * padWidth - dilationWidth * (kernelWidth - 1) - 1) / strideWidth) + 1
        
        return outputHeight to outputWidth
    }
}