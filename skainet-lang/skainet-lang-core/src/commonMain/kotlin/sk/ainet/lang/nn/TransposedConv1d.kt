package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.topology.bias
import sk.ainet.lang.nn.topology.weights

/**
 * 1D Transposed Convolutional layer (also known as fractionally-strided convolution).
 *
 * This layer is commonly used for upsampling in generative models, audio synthesis
 * (e.g. BigVGAN, Voxtral codec), and decoder architectures.
 *
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels
 * @param kernelSize Size of the convolving kernel
 * @param stride Stride of the convolution (default: 1)
 * @param padding Padding added to both sides of the input (default: 0)
 * @param outputPadding Additional size added to one side of the output (default: 0)
 * @param dilation Spacing between kernel elements (default: 1)
 * @param groups Number of blocked connections from input channels to output channels (default: 1)
 * @param bias Whether to add a learnable bias to the output (default: true)
 * @param name Name of the module
 * @param initWeights Initial weights tensor
 * @param initBias Initial bias tensor (if bias is true)
 */
public class TransposedConv1d<T : DType, V>(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Int,
    public val stride: Int = 1,
    public val padding: Int = 0,
    public val outputPadding: Int = 0,
    public val dilation: Int = 1,
    public val groups: Int = 1,
    public val bias: Boolean = true,
    override val name: String = "TransposedConv1d",
    initWeights: Tensor<T, V>,
    initBias: Tensor<T, V>? = null,
    public val trainable: Boolean = true
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        require(inChannels > 0) { "inChannels must be positive" }
        require(outChannels > 0) { "outChannels must be positive" }
        require(kernelSize > 0) { "kernelSize must be positive" }
        require(stride > 0) { "stride must be positive" }
        require(padding >= 0) { "padding must be non-negative" }
        require(outputPadding >= 0) { "outputPadding must be non-negative" }
        require(outputPadding < stride) { "outputPadding must be less than stride" }
        require(dilation > 0) { "dilation must be positive" }
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

            input.ops.convTranspose1d(
                input = input,
                weight = weight,
                bias = biasValue,
                stride = stride,
                padding = padding,
                outputPadding = outputPadding,
                dilation = dilation,
                groups = groups
            )
        }

    /**
     * Calculates the output size for a given input size and transposed convolution parameters.
     */
    public fun outputSize(inputSize: Int): Int {
        return (inputSize - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1
    }
}
