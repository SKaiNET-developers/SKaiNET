package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters

/**
 * Transposed Convolution (Deconvolution) layer.
 * 
 * Transposed convolution is used for upsampling, essentially performing the reverse
 * operation of convolution. It's commonly used in:
 * - Generative models (GANs, VAEs)
 * - Semantic segmentation (decoder networks)
 * - Super-resolution networks
 * - Any architecture requiring learned upsampling
 * 
 * Despite sometimes being called "deconvolution", it doesn't truly invert the
 * convolution operation, but rather learns an upsampling transformation.
 * 
 * The transposed convolution increases the spatial dimensions of the input,
 * making it useful for tasks that need to go from low-resolution feature maps
 * to higher-resolution outputs.
 * 
 * @param inChannels Number of input channels
 * @param outChannels Number of output channels/filters
 * @param kernelSize Size of the convolving kernel (height, width)
 * @param stride Stride of the transposed convolution (default: 1, 1)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param outputPadding Additional padding added to one side of output shape (default: 0, 0)
 * @param dilation Spacing between kernel elements (default: 1, 1)
 * @param groups Number of groups for grouped transposed convolution (default: 1)
 * @param bias Whether to add a learnable bias to the output (default: true)
 * @param name Name of the module
 * @param initWeights Initial weights tensor
 * @param initBias Initial bias tensor (if bias is true)
 */
public class TransposedConv2d<T : DType, V>(
    public val inChannels: Int,
    public val outChannels: Int,
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = 1 to 1,
    public val padding: Pair<Int, Int> = 0 to 0,
    public val outputPadding: Pair<Int, Int> = 0 to 0,
    public val dilation: Pair<Int, Int> = 1 to 1,
    public val groups: Int = 1,
    public val bias: Boolean = true,
    override val name: String = "TransposedConv2d",
    initWeights: Tensor<T, V>,
    initBias: Tensor<T, V>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {
    
    init {
        require(inChannels > 0) { "inChannels must be positive" }
        require(outChannels > 0) { "outChannels must be positive" }
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
        require(outputPadding.first >= 0 && outputPadding.second >= 0) { "outputPadding must be non-negative" }
        require(dilation.first > 0 && dilation.second > 0) { "dilation must be positive" }
        require(groups > 0) { "groups must be positive" }
        require(inChannels % groups == 0) { "inChannels must be divisible by groups" }
        require(outChannels % groups == 0) { "outChannels must be divisible by groups" }
        require(outputPadding.first < stride.first) { "outputPadding.first must be less than stride.first" }
        require(outputPadding.second < stride.second) { "outputPadding.second must be less than stride.second" }
    }

    override val params: List<ModuleParameter<T, V>> = buildList {
        add(ModuleParameter.WeightParameter("$name.weight", initWeights))
        if (bias && initBias != null) {
            add(ModuleParameter.BiasParameter("$name.bias", initBias))
        }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val weight = params.first { it.name.endsWith(".weight") }.value
            val biasValue = if (bias) params.firstOrNull { it.name.endsWith(".bias") }?.value else null
            input.ops.convTranspose2d(input, weight, biasValue, stride, padding, outputPadding, dilation, groups)
        }

    /**
     * Calculates the output size for a given input size and transposed convolution parameters.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        val (inputHeight, inputWidth) = inputSize
        val (kernelHeight, kernelWidth) = kernelSize
        val (strideHeight, strideWidth) = stride
        val (padHeight, padWidth) = padding
        val (outputPadHeight, outputPadWidth) = outputPadding
        val (dilationHeight, dilationWidth) = dilation
        
        // Transposed convolution output size formula
        val outputHeight = (inputHeight - 1) * strideHeight - 2 * padHeight + 
                          dilationHeight * (kernelHeight - 1) + outputPadHeight + 1
        val outputWidth = (inputWidth - 1) * strideWidth - 2 * padWidth + 
                         dilationWidth * (kernelWidth - 1) + outputPadWidth + 1
        
        return outputHeight to outputWidth
    }

    /**
     * Returns the total number of parameters in this layer.
     */
    public fun parameterCount(): Long {
        val weightParams = (inChannels / groups) * (outChannels / groups) * kernelSize.first * kernelSize.second * groups
        val biasParams = if (bias) outChannels else 0
        return (weightParams + biasParams).toLong()
    }

    /**
     * Returns the upsampling factor based on stride.
     */
    public fun upsamplingFactor(): Pair<Double, Double> {
        return stride.first.toDouble() to stride.second.toDouble()
    }

    /**
     * Checks if this is a simple 2x upsampling (stride == (2, 2)).
     */
    public fun is2xUpsampling(): Boolean = stride == 2 to 2

    /**
     * Checks if this is a standard transposed convolution (no dilation, no groups).
     */
    public fun isStandard(): Boolean = dilation == 1 to 1 && groups == 1

    /**
     * Returns information about the transposed convolution configuration.
     */
    public fun transposedConvInfo(): TransposedConvInfo {
        return TransposedConvInfo(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            outputPadding = outputPadding,
            dilation = dilation,
            groups = groups,
            upsamplingFactor = upsamplingFactor(),
            parameterCount = parameterCount(),
            is2xUpsampling = is2xUpsampling(),
            isStandard = isStandard()
        )
    }

    /**
     * Factory methods for common transposed convolution configurations.
     */
    public companion object {
        /**
         * Creates a standard 2x upsampling transposed convolution.
         */
        public fun <T : DType, V> upsampling2x(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Pair<Int, Int> = 4 to 4,
            padding: Pair<Int, Int> = 1 to 1,
            bias: Boolean = true,
            name: String = "TransposedConv2d_2x",
            initWeights: Tensor<T, V>,
            initBias: Tensor<T, V>? = null
        ): TransposedConv2d<T, V> {
            return TransposedConv2d<T, V>(
                inChannels = inChannels,
                outChannels = outChannels,
                kernelSize = kernelSize,
                stride = 2 to 2,
                padding = padding,
                bias = bias,
                name = name,
                initWeights = initWeights,
                initBias = initBias
            )
        }

        /**
         * Creates a transposed convolution for custom upsampling factor.
         */
        public fun <T : DType, V> customUpsampling(
            inChannels: Int,
            outChannels: Int,
            upsamplingFactor: Pair<Int, Int>,
            kernelSize: Pair<Int, Int>? = null,
            padding: Pair<Int, Int>? = null,
            bias: Boolean = true,
            name: String = "TransposedConv2d_custom",
            initWeights: Tensor<T, V>,
            initBias: Tensor<T, V>? = null
        ): TransposedConv2d<T, V> {
            // Default kernel size and padding for clean upsampling
            val defaultKernelSize = kernelSize ?: (upsamplingFactor.first * 2 to upsamplingFactor.second * 2)
            val defaultPadding = padding ?: (upsamplingFactor.first / 2 to upsamplingFactor.second / 2)
            
            return TransposedConv2d<T, V>(
                inChannels = inChannels,
                outChannels = outChannels,
                kernelSize = defaultKernelSize,
                stride = upsamplingFactor,
                padding = defaultPadding,
                bias = bias,
                name = name,
                initWeights = initWeights,
                initBias = initBias
            )
        }
    }
}

/**
 * Data class containing information about transposed convolution configuration.
 */
public data class TransposedConvInfo(
    val kernelSize: Pair<Int, Int>,
    val stride: Pair<Int, Int>,
    val padding: Pair<Int, Int>,
    val outputPadding: Pair<Int, Int>,
    val dilation: Pair<Int, Int>,
    val groups: Int,
    val upsamplingFactor: Pair<Double, Double>,
    val parameterCount: Long,
    val is2xUpsampling: Boolean,
    val isStandard: Boolean
)