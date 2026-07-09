package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.activations.ActivationsWrapperModule
import sk.ainet.lang.nn.AvgPool2d
import sk.ainet.lang.nn.Conv1d
import sk.ainet.lang.nn.Conv2d
import sk.ainet.lang.nn.Conv3d
import sk.ainet.lang.nn.Flatten
import sk.ainet.lang.nn.Gru
import sk.ainet.lang.nn.Input
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.MaxPool2d
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.Upsample2d
import sk.ainet.lang.nn.normalization.BatchNormalization
import sk.ainet.lang.nn.normalization.GroupNormalization
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.ConvShapeUtils
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.dsl.TensorCreationScope
import sk.ainet.lang.nn.activations.Softmax
import kotlin.reflect.KClass

// DSL Marker to restrict the DSL to its intended scope
@DslMarker
public annotation class NetworkDsl

/**
 * Generic network builder function that creates a neural network with specified data type and value type.
 *
 * @param T The data type (DType) - must extend DType (e.g., FP32, FP16, Int8, Int32, Ternary, Int4)
 * @param V The value type - must match the DType's native type:
 *   - FP32 → Float
 *   - FP16 → Float (promoted)
 *   - Int32 → Int
 *   - Int8 → Byte
 *   - Int4 → Byte (promoted)
 *   - Ternary → Byte (special case)
 * @param content The DSL content block that defines the network structure
 * @return A Module<T, V> representing the complete neural network
 *
 * Example usage:
 * ```kotlin
 * val fpNetwork = network<FP32, Float> {
 *     input(784)
 *     dense(128) { weights { shape -> CpuTensorFP32.random(shape) } }
 *     dense(10) { weights { shape -> CpuTensorFP32.random(shape) } }
 * }
 *
 * val intNetwork = network<Int8, Byte> {
 *     input(28)
 *     dense(16) { weights { shape -> CpuTensorInt8.ones(shape) } }
 * }
 * ```
 */
@NetworkDsl
public inline fun <reified T : DType, V> sequential(
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> =
    NeuralNetworkDslImpl<T, V>(DefaultNeuralNetworkExecutionContext(), T::class)
        .apply(content)
        .create()

/**
 * Overload that wires both tensor factory and ops from an ExecutionContext.
 */
public inline fun <reified T : DType, V> sequential(
    executionContext: ExecutionContext,
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> =
    NeuralNetworkDslImpl<T, V>(executionContext, T::class)
        .apply(content)
        .create()

@NetworkDsl
public interface NetworkDslItem {
    public val executionContext: ExecutionContext
}

/**
 * Core DSL interface for building neural networks with generic tensor types.
 * This interface provides a fluent API for constructing neural network architectures
 * with support for different data types and precision levels.
 *
 * @param T The data type (DType) that determines the precision and storage format
 * @param V The value type that corresponds to the native Kotlin type for the DType
 *
 * Type constraints ensure compatibility between DType and value type:
 * - T must extend DType to ensure valid tensor operations
 * - V should match the native type expected by the DType implementation
 *
 * Performance considerations:
 * - FP32/Float: Best accuracy, higher memory usage
 * - FP16/Float: Reduced memory, slightly lower accuracy
 * - Int8/Byte: Minimal memory, quantized operations
 * - Int32/Int: Integer operations, specific use cases
 */
@NetworkDsl
public interface NeuralNetworkDsl<T : DType, V> : NetworkDslItem {
    /**
     * Creates an input layer that defines the entry point for data into the network.
     *
     * @param inputSize The number of input features/dimensions
     * @param id Optional identifier for the layer (auto-generated if empty)
     * @param requiresGrad Whether the input requires gradients (default: false)
     */
    public fun input(inputSize: Int, id: String = "", requiresGrad: Boolean = false)

    /**
     * Declares a multi-dimensional input shape (per-sample, batch dimension excluded).
     *
     * Required when downstream spatial layers (conv, pool, upsample) need to feed
     * a `flatten()` -> `dense()` chain — without this, the DSL cannot know the
     * flattened feature count ahead of time.
     *
     * Example: `input(intArrayOf(1, 28, 28))` declares one-channel 28x28 images.
     *
     * @param inputShape Per-sample shape (e.g. `[channels, height, width]`)
     * @param id Optional identifier for the layer
     * @param requiresGrad Whether the input requires gradients (default: false)
     */
    public fun input(inputShape: IntArray, id: String = "", requiresGrad: Boolean = false): Unit =
        input(inputShape.fold(1) { a, b -> a * b }, id, requiresGrad)

    /**
     * Creates a flatten layer that reshapes multidimensional tensors into 1D.
     * Useful for transitioning from convolutional to dense layers.
     *
     * @param id Optional identifier for the layer
     * @param content Configuration block for flatten-specific parameters
     */
    public fun flatten(id: String = "", content: FLATTEN<T, V>.() -> Unit = {})

    /**
     * Creates a dense (fully connected) layer with specified output dimension.
     *
     * @param outputDimension The number of neurons/output features
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights, bias, and activation
     */
    public fun dense(outputDimension: Int, id: String = "", content: DENSE<T, V>.() -> Unit = {})

    /**
     * Creates a dense layer without specifying output dimension (must be set in content block).
     *
     * @param id Optional identifier for the layer
     * @param content Configuration block where output dimension, weights, and bias are set
     */
    public fun dense(id: String = "", content: DENSE<T, V>.() -> Unit = {})

    /**
     * Creates a single-layer unidirectional GRU. Input `[batch, seq, features]` ->
     * output `[batch, seq, hiddenSize]`. Input feature size is inferred from the
     * preceding layer's output dimension.
     *
     * @param hiddenSize size of the GRU hidden state
     * @param id Optional identifier for the layer
     * @param content Configuration block (e.g. `trainable`)
     */
    public fun gru(hiddenSize: Int, id: String = "", content: GRU<T, V>.() -> Unit = {})

    /**
     * Creates a dense layer with precision override and specified output dimension.
     * This allows individual layers to use different precision than the network default.
     *
     * @param TLayer The precision type for this specific layer
     * @param outputDimension The number of neurons/output features
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights, bias, and activation
     */
    public fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String = "",
        content: DENSE<TLayer, V>.() -> Unit = {}
    ): Module<T, V>

    /**
     * Creates a dense layer with precision override without specifying output dimension.
     *
     * @param TLayer The precision type for this specific layer
     * @param id Optional identifier for the layer
     * @param content Configuration block where output dimension, weights, and bias are set
     */
    public fun <TLayer : DType> dense(
        id: String = "",
        content: DENSE<TLayer, V>.() -> Unit = {}
    ): Module<T, V>

    /**
     * Applies an activation function as a separate layer.
     *
     * @param id Optional identifier for the activation layer
     * @param activation Function that transforms tensor values (e.g., ReLU, Sigmoid)
     */
    public fun activation(id: String = "", activation: (Tensor<T, V>) -> Tensor<T, V>)

    /**
     * Applies a Softmax activation as a separate layer.
     *
     * @param dim Dimension along which softmax will be computed. Supports negative indexing.
     * @param id Optional identifier for the layer
     */
    public fun softmax(dim: Int = -1, id: String = "")

    /**
     * Creates a batch normalization layer for training stability and performance.
     * Normalizes the input across the batch dimension.
     *
     * @param numFeatures Number of features (channels)
     * @param eps Small value added to the denominator for numerical stability
     * @param momentum Momentum for running statistics update during training
     * @param affine Whether to learn affine parameters (gamma and beta)
     * @param id Optional identifier for the layer
     */
    public fun batchNorm(
        numFeatures: Int,
        eps: Double = 1e-5,
        momentum: Double = 0.1,
        affine: Boolean = true,
        id: String = ""
    )

    /**
     * Creates a group normalization layer - alternative normalization approach.
     * Normalizes the input by dividing channels into groups and normalizing within each group.
     *
     * @param numGroups Number of groups to divide the channels into
     * @param numChannels Number of channels in the input
     * @param eps Small value added to the denominator for numerical stability
     * @param affine Whether to learn affine parameters (gamma and beta)
     * @param id Optional identifier for the layer
     */
    public fun groupNorm(
        numGroups: Int,
        numChannels: Int,
        eps: Double = 1e-5,
        affine: Boolean = true,
        id: String = ""
    )

    /**
     * Creates a layer normalization layer - used in attention mechanisms.
     * Normalizes the input across the last dimension(s).
     *
     * @param normalizedShape The shape of the normalization (typically the last dimension(s))
     * @param eps Small value added to the denominator for numerical stability
     * @param elementwiseAffine Whether to learn elementwise affine parameters (gamma and beta)
     * @param id Optional identifier for the layer
     */
    public fun layerNorm(
        normalizedShape: IntArray,
        eps: Double = 1e-5,
        elementwiseAffine: Boolean = true,
        id: String = ""
    )

    /**
     * Creates a 2D convolutional layer for processing spatial data like images.
     *
     * @param outChannels Number of output channels/filters
     * @param kernelSize Size of the convolving kernel (height, width)
     * @param stride Stride of the convolution (default: 1, 1)
     * @param padding Padding added to all sides of the input (default: 0, 0)
     * @param dilation Spacing between kernel elements (default: 1, 1)
     * @param groups Number of groups for grouped convolution (default: 1)
     * @param bias Whether to add a learnable bias (default: true)
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights and bias initialization
     */
    public fun conv2d(
        outChannels: Int,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1,
        bias: Boolean = true,
        id: String = "",
        content: CONV2D<T, V>.() -> Unit = {}
    )

    /**
     * Creates a 2D convolutional layer with all parameters configured inside the DSL block.
     * Example:
     * conv2d("conv1") {
     *     outChannels = 16
     *     kernelSize(5)
     *     stride(1)
     *     padding(2)
     * }
     */
    public fun conv2d(
        id: String = "",
        content: CONV2D<T, V>.() -> Unit
    )

    /**
     * Creates a 2D max pooling layer for downsampling feature maps.
     *
     * @param kernelSize Size of the pooling window (height, width)
     * @param stride Stride of the pooling operation (default: same as kernelSize)
     * @param padding Padding added to all sides of the input (default: 0, 0)
     * @param id Optional identifier for the layer
     */
    public fun maxPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0,
        id: String = ""
    )

    /**
     * Creates a 2D max pooling layer with all parameters configured inside the DSL block.
     * Example:
     * maxPool2d("pool1") {
     *     kernelSize(2)
     *     stride(2)
     *     padding(0)
     * }
     */
    public fun maxPool2d(
        id: String = "",
        content: MAXPOOL2D<T, V>.() -> Unit
    )

    /**
     * Creates a 2D upsampling layer for increasing spatial resolution.
     *
     * @param scale Upsampling factors for height and width
     * @param mode Interpolation mode (nearest default)
     * @param alignCorners Alignment flag for bilinear mode (ignored for nearest)
     * @param id Optional identifier for the layer
     */
    public fun upsample2d(
        scale: Pair<Int, Int> = 2 to 2,
        mode: UpsampleMode = UpsampleMode.Nearest,
        alignCorners: Boolean = false,
        id: String = ""
    )

    /**
     * Creates a 2D upsampling layer with parameters configured in the DSL block.
     */
    public fun upsample2d(
        id: String = "",
        content: UPSAMPLE2D<T, V>.() -> Unit
    )

    /**
     * Creates a 1D convolutional layer for processing sequence data.
     *
     * @param outChannels Number of output channels/filters
     * @param kernelSize Size of the convolving kernel
     * @param stride Stride of the convolution (default: 1)
     * @param padding Padding added to both sides of the input (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of groups for grouped convolution (default: 1)
     * @param bias Whether to add a learnable bias (default: true)
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights and bias initialization
     */
    public fun conv1d(
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Boolean = true,
        id: String = "",
        content: CONV1D<T, V>.() -> Unit = {}
    )

    /**
     * Creates a 3D convolutional layer for processing volumetric data.
     *
     * @param outChannels Number of output channels/filters
     * @param kernelSize Size of the convolving kernel (depth, height, width)
     * @param stride Stride of the convolution (default: 1, 1, 1)
     * @param padding Padding added to all sides of the input (default: 0, 0, 0)
     * @param dilation Spacing between kernel elements (default: 1, 1, 1)
     * @param groups Number of groups for grouped convolution (default: 1)
     * @param bias Whether to add a learnable bias (default: true)
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights and bias initialization
     */
    public fun conv3d(
        outChannels: Int,
        kernelSize: Triple<Int, Int, Int>,
        stride: Triple<Int, Int, Int> = Triple(1, 1, 1),
        padding: Triple<Int, Int, Int> = Triple(0, 0, 0),
        dilation: Triple<Int, Int, Int> = Triple(1, 1, 1),
        groups: Int = 1,
        bias: Boolean = true,
        id: String = "",
        content: CONV3D<T, V>.() -> Unit = {}
    )

    /**
     * Creates a 2D average pooling layer for downsampling feature maps.
     *
     * @param kernelSize Size of the pooling window (height, width)
     * @param stride Stride of the pooling operation (default: same as kernelSize)
     * @param padding Padding added to all sides of the input (default: 0, 0)
     * @param countIncludePad Whether to include padding in the average calculation (default: true)
     * @param id Optional identifier for the layer
     */
    public fun avgPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0,
        countIncludePad: Boolean = true,
        id: String = ""
    )

    // LLM / Transformer layers (embedding, rmsNorm, multiHeadAttention, swiGluFFN,
    // residual, xielu) have been moved to skainet-transformers llm-core module.

    /**
     * Groups layers into a sequential block for better organization.
     *
     * @param content DSL block containing the sequence of layers
     */
    public fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit)

    /**
     * Creates a named stage/block within the network for modular design.
     *
     * @param id Identifier for the stage
     * @param content DSL block containing the layers within this stage
     */
    public fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit)

    /**
     * Creates a precision-scoped stage within the network.
     * This allows grouping layers with a specific precision type that differs
     * from the network default, enabling fine-grained mixed-precision control.
     *
     * Unlike the basic stage method, this version allows changing the precision
     * type for all layers within the stage scope.
     *
     * @param TStage The precision type for all layers within this stage
     * @param id Identifier for the stage
     * @param content DSL block containing layers with TStage precision
     * @return Module that handles the precision conversion automatically
     */
    public fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V>
}

public interface WandBTensorValueContext<T : DType, V> {
    public val executionContext: ExecutionContext
    public val weightsShape: Shape
    public val biasShape: Shape

    public fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>)
    public fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>)
}

@NetworkDsl
public interface DENSE<T : DType, V> : NetworkDslItem, WandBTensorValueContext<T, V> {
    public var activation: (Tensor<T, V>) -> Tensor<T, V>
    public var units: Int
    public var trainable: Boolean
}

@NetworkDsl
public interface CONV2D<T : DType, V> : NetworkDslItem, WandBTensorValueContext<T, V> {
    public var inChannels: Int
    public var outChannels: Int
    public var kernelSize: Pair<Int, Int>
    public var stride: Pair<Int, Int>
    public var padding: Pair<Int, Int>
    public var dilation: Pair<Int, Int>
    public var groups: Int
    public var bias: Boolean
    public var trainable: Boolean

    // Helper setters to allow concise Int-based configuration in DSL blocks
    public fun kernelSize(size: Int)
    public fun stride(size: Int)
    public fun padding(size: Int)
}

@NetworkDsl
public interface MAXPOOL2D<T : DType, V> : NetworkDslItem {
    public var kernelSize: Pair<Int, Int>
    public var stride: Pair<Int, Int>
    public var padding: Pair<Int, Int>

    // Helper setters to allow concise Int-based configuration in DSL blocks
    public fun kernelSize(size: Int)
    public fun stride(size: Int)
    public fun padding(size: Int)
}

@NetworkDsl
public interface UPSAMPLE2D<T : DType, V> : NetworkDslItem {
    public var scale: Pair<Int, Int>
    public var mode: UpsampleMode
    public var alignCorners: Boolean

    // Helper setter to allow concise Int-based configuration in DSL blocks
    public fun scale(factor: Int)
}

@NetworkDsl
public interface CONV1D<T : DType, V> : NetworkDslItem, WandBTensorValueContext<T, V> {
    public var inChannels: Int
    public var outChannels: Int
    public var kernelSize: Int
    public var stride: Int
    public var padding: Int
    public var dilation: Int
    public var groups: Int
    public var bias: Boolean
    public var trainable: Boolean
}

@NetworkDsl
public interface CONV3D<T : DType, V> : NetworkDslItem, WandBTensorValueContext<T, V> {
    public var inChannels: Int
    public var outChannels: Int
    public var kernelSize: Triple<Int, Int, Int>
    public var stride: Triple<Int, Int, Int>
    public var padding: Triple<Int, Int, Int>
    public var dilation: Triple<Int, Int, Int>
    public var groups: Int
    public var bias: Boolean
    public var trainable: Boolean

    // Helper setters to allow concise Int-based configuration in DSL blocks
    public fun kernelSize(size: Int)
    public fun stride(size: Int)
    public fun padding(size: Int)
}

// ATTENTION interface moved to skainet-transformers llm-core TransformerDsl.kt

@NetworkDsl
public interface AVGPOOL2D<T : DType, V> : NetworkDslItem {
    public var kernelSize: Pair<Int, Int>
    public var stride: Pair<Int, Int>
    public var padding: Pair<Int, Int>
    public var countIncludePad: Boolean

    // Helper setters to allow concise Int-based configuration in DSL blocks
    public fun kernelSize(size: Int)
    public fun stride(size: Int)
    public fun padding(size: Int)
}

/**
 * Scope for weights initialization with implicit shape context.
 */
@NetworkDsl
public interface WeightsScope<T : DType, V> : TensorCreationScope<T, V>

/**
 * Scope for bias initialization with implicit shape context.
 */
@NetworkDsl
public interface BiasScope<T : DType, V> : TensorCreationScope<T, V>

/**
 * Implementation of WeightsScope for weights initialization.
 */
public class WeightsScopeImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    override val shape: Shape,
    override val dtype: KClass<T>
) : WeightsScope<T, V>

/**
 * Implementation of BiasScope for bias initialization.
 */
public class BiasScopeImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    override val shape: Shape,
    override val dtype: KClass<T>,
) : BiasScope<T, V>

@NetworkDsl
public interface FLATTEN<T : DType, V> : NetworkDslItem {
    public var startDim: Int
    public var endDim: Int
}

public fun getDefaultName(id: String, s: String, size: Int): String {
    if (id.isNotEmpty()) return id
    return "$s-$size"
}

public class FlattenImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    override var startDim: Int = 1,
    override var endDim: Int = -1,
    private val id: String,
) : FLATTEN<T, V> {
    public fun create(): Module<T, V> {
        return Flatten(startDim, endDim, id)
    }
}

/** DSL config handle for a [Gru] layer. */
public interface GRU<T : DType, V> : NetworkDslItem {
    public var trainable: Boolean
}

public class GruImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val id: String,
    private val kClass: KClass<T>,
) : GRU<T, V> {
    override var trainable: Boolean = true

    public fun create(): Gru<T, V> {
        require(inputSize > 0) { "Gru inputSize must be > 0 (declare an input shape before gru)." }
        require(hiddenSize > 0) { "Gru hiddenSize must be > 0." }
        val g = 3 * hiddenSize
        return Gru(
            inputSize = inputSize,
            hiddenSize = hiddenSize,
            name = id,
            initWeightIh = executionContext.placeholder(Shape(inputSize, g), kClass),
            initWeightHh = executionContext.placeholder(Shape(hiddenSize, g), kClass),
            initBiasIh = executionContext.placeholder(Shape(g), kClass),
            initBiasHh = executionContext.placeholder(Shape(g), kClass),
            trainable = trainable,
        )
    }
}

private fun <T : DType, V> createLinear(
    executionContext: ExecutionContext,
    inFeatures: Int,
    outFeatures: Int,
    id: String,
    kClass: KClass<T>,
    myInitWeights: Tensor<T, V>? = null,
    myInitBias: Tensor<T, V>? = null,
    trainable: Boolean = true
): Linear<T, V> {
    return when {
        myInitWeights != null && myInitBias != null ->
            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = myInitBias,
                trainable = trainable
            )

        myInitWeights == null && myInitBias != null -> {

            val safeWeights = executionContext.tensorDataFactory.placeholder<T, V>(Shape(outFeatures, inFeatures), kClass)
            val initW = executionContext.fromData(safeWeights, kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = initW,
                initBias = myInitBias,
                trainable = trainable
            )
        }

        myInitWeights != null && myInitBias == null -> {
            val safeBias = executionContext.tensorDataFactory.placeholder<T, V>(Shape(outFeatures), kClass)
            val initB = executionContext.fromData(safeBias, kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = initB,
                trainable = trainable
            )
        }

        else -> {
            val safeWeights = executionContext.tensorDataFactory.placeholder<T, V>(Shape(outFeatures, inFeatures), kClass)
            val safeBias = executionContext.tensorDataFactory.placeholder<T, V>(Shape(outFeatures), kClass)
            val initW = executionContext.fromData(safeWeights, kClass)
            val initB = executionContext.fromData(safeBias, kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = initW,
                initBias = initB,
                trainable = trainable
            )
        }
    }
}


public class DenseImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    private val inputDimension: Int,
    private var _outputDimension: Int,
    private val id: String,
    private val kClass: KClass<T>,

    ) : DENSE<T, V> {

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null
    private var _activation: (Tensor<T, V>) -> Tensor<T, V> = { tensor -> tensor }
    override var trainable: Boolean = true

    // Expose the output dimension
    public val outputDimension: Int
        get() = _outputDimension

    // Shape context for the DSL
    override val weightsShape: Shape
        get() = Shape(_outputDimension, inputDimension)

    override val biasShape: Shape
        get() = Shape(_outputDimension)

    private var activationSet: Boolean = false

    public fun create(): List<Module<T, V>> {
        // Create default tensors if not provided - use factory for defaults
        val weights = weightsValue
        val bias = biasValue

        val linear = createLinear(
            inFeatures = inputDimension,
            outFeatures = _outputDimension,
            id = id,
            kClass = kClass,
            myInitWeights = weights,
            myInitBias = bias,
            executionContext = executionContext,
            trainable = trainable
        )

        // Build module list: always the linear layer, and optionally the activation if set inside dense{}
        val modules = mutableListOf<Module<T, V>>()
        modules += linear
        if (activationSet) {
            modules += ActivationsWrapperModule(_activation, getDefaultName("$id-activation", "activation", 0))
        }
        return modules
    }

    override var activation: (Tensor<T, V>) -> Tensor<T, V>
        get() = _activation
        set(value) {
            _activation = value
            activationSet = true
        }

    override var units: Int
        get() = _outputDimension
        set(value) {
            _outputDimension = value
        }


    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = WeightsScopeImpl<T, V>(executionContext, weightsShape, kClass)
        weightsValue = scope.initBlock(weightsShape)
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = BiasScopeImpl<T, V>(executionContext, biasShape, kClass)
        biasValue = scope.initBlock(biasShape)
    }
}

public class Conv2dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialInChannels: Int,
    initialOutChannels: Int,
    initialKernelSize: Pair<Int, Int>,
    initialStride: Pair<Int, Int>,
    initialPadding: Pair<Int, Int>,
    initialDilation: Pair<Int, Int>,
    initialGroups: Int,
    initialBias: Boolean,
    private val id: String,
    private val kClass: KClass<T>,

    ) : CONV2D<T, V> {

    // Helper setters implementation
    override fun kernelSize(size: Int) { this.kernelSize = size to size }
    override fun stride(size: Int) { this.stride = size to size }
    override fun padding(size: Int) { this.padding = size to size }

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null

    // Override mutable properties from CONV2D interface
    override var inChannels: Int = initialInChannels
    override var outChannels: Int = initialOutChannels
    override var kernelSize: Pair<Int, Int> = initialKernelSize
    override var stride: Pair<Int, Int> = initialStride
    override var padding: Pair<Int, Int> = initialPadding
    override var dilation: Pair<Int, Int> = initialDilation
    override var groups: Int = initialGroups
    override var bias: Boolean = initialBias
    override var trainable: Boolean = true

    // Shape context for the DSL
    override val weightsShape: Shape
        get() = Shape(intArrayOf(outChannels, inChannels, kernelSize.first, kernelSize.second))

    override val biasShape: Shape
        get() = Shape(intArrayOf(outChannels))

    public fun create(): Conv2d<T, V> {
        // Validate required fields
        require(outChannels > 0) { "Conv2d outChannels must be > 0. Set it in the DSL block." }
        require(kernelSize.first > 0 && kernelSize.second > 0) { "Conv2d kernelSize must be > 0. Set it in the DSL block." }
        require(inChannels > 0) { "Conv2d inChannels must be > 0 (set explicitly if not inferred)." }

        // Create default tensors if not provided
        val weights = weightsValue ?: executionContext.placeholder(weightsShape, kClass)

        val biasParam = if (bias) {
            biasValue ?: executionContext.placeholder(biasShape, kClass)
        } else null

        return Conv2d(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            name = getDefaultName(id, "Conv2d", 0),
            initWeights = weights,
            initBias = biasParam,
            trainable = trainable
        )
    }

    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = WeightsScopeImpl<T, V>(executionContext, weightsShape, kClass)
        weightsValue = scope.initBlock(weightsShape)
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = BiasScopeImpl<T, V>(executionContext, biasShape, kClass)
        biasValue = scope.initBlock(biasShape)
    }
}

public class MaxPool2dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialKernelSize: Pair<Int, Int>,
    initialStride: Pair<Int, Int>,
    initialPadding: Pair<Int, Int>,
    private val id: String,
) : MAXPOOL2D<T, V> {

    // Helper setters implementation
    override fun kernelSize(size: Int) { this.kernelSize = size to size }
    override fun stride(size: Int) { this.stride = size to size }
    override fun padding(size: Int) { this.padding = size to size }

    override var kernelSize: Pair<Int, Int> = initialKernelSize
    override var stride: Pair<Int, Int> = initialStride
    override var padding: Pair<Int, Int> = initialPadding

    public fun create(): MaxPool2d<T, V> {
        require(kernelSize.first > 0 && kernelSize.second > 0) { "MaxPool2d kernelSize must be > 0. Set it in the DSL block." }
        require(stride.first > 0 && stride.second > 0) { "MaxPool2d stride must be > 0. Set it in the DSL block." }
        require(padding.first >= 0 && padding.second >= 0) { "MaxPool2d padding must be >= 0. Set it in the DSL block." }

        return MaxPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            name = getDefaultName(id, "MaxPool2d", 0)
        )
    }
}

public class Upsample2dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialScale: Pair<Int, Int>,
    initialMode: UpsampleMode,
    initialAlignCorners: Boolean,
    private val id: String,
) : UPSAMPLE2D<T, V> {

    override var scale: Pair<Int, Int> = initialScale
    override var mode: UpsampleMode = initialMode
    override var alignCorners: Boolean = initialAlignCorners

    override fun scale(factor: Int) {
        this.scale = factor to factor
    }

    public fun create(): Upsample2d<T, V> {
        require(scale.first > 0 && scale.second > 0) { "Upsample2d scale must be > 0. Set it in the DSL block." }
        return Upsample2d(
            scale = scale,
            mode = mode,
            alignCorners = alignCorners,
            name = getDefaultName(id, "Upsample2d", 0)
        )
    }
}

public class Conv1dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialInChannels: Int,
    initialOutChannels: Int,
    initialKernelSize: Int,
    initialStride: Int,
    initialPadding: Int,
    initialDilation: Int,
    initialGroups: Int,
    initialBias: Boolean,
    private val id: String,
    private val kClass: KClass<T>,
) : CONV1D<T, V> {

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null

    override var inChannels: Int = initialInChannels
    override var outChannels: Int = initialOutChannels
    override var kernelSize: Int = initialKernelSize
    override var stride: Int = initialStride
    override var padding: Int = initialPadding
    override var dilation: Int = initialDilation
    override var groups: Int = initialGroups
    override var bias: Boolean = initialBias
    override var trainable: Boolean = true

    override val weightsShape: Shape
        get() = Shape(intArrayOf(outChannels, inChannels / groups, kernelSize))

    override val biasShape: Shape
        get() = Shape(intArrayOf(outChannels))

    public fun create(): Conv1d<T, V> {
        require(outChannels > 0) { "Conv1d outChannels must be > 0." }
        require(kernelSize > 0) { "Conv1d kernelSize must be > 0." }
        require(inChannels > 0) { "Conv1d inChannels must be > 0." }

        val weights = weightsValue ?: executionContext.placeholder(weightsShape, kClass)
        val biasParam = if (bias) biasValue ?: executionContext.placeholder(biasShape, kClass) else null

        return Conv1d(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            name = getDefaultName(id, "Conv1d", 0),
            initWeights = weights,
            initBias = biasParam,
            trainable = trainable
        )
    }

    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = WeightsScopeImpl<T, V>(executionContext, weightsShape, kClass)
        weightsValue = scope.initBlock(weightsShape)
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = BiasScopeImpl<T, V>(executionContext, biasShape, kClass)
        biasValue = scope.initBlock(biasShape)
    }
}

public class Conv3dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialInChannels: Int,
    initialOutChannels: Int,
    initialKernelSize: Triple<Int, Int, Int>,
    initialStride: Triple<Int, Int, Int>,
    initialPadding: Triple<Int, Int, Int>,
    initialDilation: Triple<Int, Int, Int>,
    initialGroups: Int,
    initialBias: Boolean,
    private val id: String,
    private val kClass: KClass<T>,
) : CONV3D<T, V> {

    override fun kernelSize(size: Int) { this.kernelSize = Triple(size, size, size) }
    override fun stride(size: Int) { this.stride = Triple(size, size, size) }
    override fun padding(size: Int) { this.padding = Triple(size, size, size) }

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null

    override var inChannels: Int = initialInChannels
    override var outChannels: Int = initialOutChannels
    override var kernelSize: Triple<Int, Int, Int> = initialKernelSize
    override var stride: Triple<Int, Int, Int> = initialStride
    override var padding: Triple<Int, Int, Int> = initialPadding
    override var dilation: Triple<Int, Int, Int> = initialDilation
    override var groups: Int = initialGroups
    override var bias: Boolean = initialBias
    override var trainable: Boolean = true

    override val weightsShape: Shape
        get() = Shape(intArrayOf(outChannels, inChannels / groups, kernelSize.first, kernelSize.second, kernelSize.third))

    override val biasShape: Shape
        get() = Shape(intArrayOf(outChannels))

    public fun create(): Conv3d<T, V> {
        require(outChannels > 0) { "Conv3d outChannels must be > 0." }
        require(kernelSize.first > 0 && kernelSize.second > 0 && kernelSize.third > 0) { "Conv3d kernelSize must be > 0." }
        require(inChannels > 0) { "Conv3d inChannels must be > 0." }

        val weights = weightsValue ?: executionContext.placeholder(weightsShape, kClass)
        val biasParam = if (bias) biasValue ?: executionContext.placeholder(biasShape, kClass) else null

        return Conv3d(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            name = getDefaultName(id, "Conv3d", 0),
            initWeights = weights,
            initBias = biasParam,
            trainable = trainable
        )
    }

    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = WeightsScopeImpl<T, V>(executionContext, weightsShape, kClass)
        weightsValue = scope.initBlock(weightsShape)
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = BiasScopeImpl<T, V>(executionContext, biasShape, kClass)
        biasValue = scope.initBlock(biasShape)
    }
}

public class AvgPool2dImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    initialKernelSize: Pair<Int, Int>,
    initialStride: Pair<Int, Int>,
    initialPadding: Pair<Int, Int>,
    initialCountIncludePad: Boolean,
    private val id: String,
) : AVGPOOL2D<T, V> {

    override fun kernelSize(size: Int) { this.kernelSize = size to size }
    override fun stride(size: Int) { this.stride = size to size }
    override fun padding(size: Int) { this.padding = size to size }

    override var kernelSize: Pair<Int, Int> = initialKernelSize
    override var stride: Pair<Int, Int> = initialStride
    override var padding: Pair<Int, Int> = initialPadding
    override var countIncludePad: Boolean = initialCountIncludePad

    public fun create(): AvgPool2d<T, V> {
        require(kernelSize.first > 0 && kernelSize.second > 0) { "AvgPool2d kernelSize must be > 0." }
        require(stride.first > 0 && stride.second > 0) { "AvgPool2d stride must be > 0." }
        require(padding.first >= 0 && padding.second >= 0) { "AvgPool2d padding must be >= 0." }

        return AvgPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            countIncludePad = countIncludePad,
            name = getDefaultName(id, "AvgPool2d", 0)
        )
    }
}

// AttentionImpl moved to skainet-transformers llm-core TransformerDsl.kt

// Stage implementation
/**
 * Computes the flattened-feature dimension for `flatten(startDim, endDim)` given a
 * known per-sample shape (batch dimension excluded).
 *
 * Tensor shape is conceptually `[batch] + currentShape`. `startDim`/`endDim` follow
 * PyTorch semantics on that full tensor shape and may be negative. Returns `null`
 * when the span doesn't include any per-sample dimension (e.g. flatten only over
 * the batch axis), which leaves the dense feature dimension unchanged.
 */
/**
 * Per-sample shape produced by a `MaxPool2d`/`AvgPool2d` given the layer's kernel/stride/padding.
 * Returns `null` when the input shape is unknown or not rank-3 `(C, H, W)`.
 */
internal fun pool2dNextShape(
    currentShape: IntArray?,
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int>,
    padding: Pair<Int, Int>
): IntArray? {
    val shape = currentShape ?: return null
    if (shape.size != 3) return null
    val out = ConvShapeUtils.pool2dOutputShape(
        intArrayOf(1, shape[0], shape[1], shape[2]),
        kernelSize, stride, padding
    )
    return intArrayOf(out[1], out[2], out[3])
}

/**
 * Per-sample shape produced by `Upsample2d` given the scale factors.
 * Returns `null` when the input shape is unknown or not rank-3 `(C, H, W)`.
 */
internal fun upsample2dNextShape(currentShape: IntArray?, scale: Pair<Int, Int>): IntArray? {
    val shape = currentShape ?: return null
    if (shape.size != 3) return null
    val out = ConvShapeUtils.upsample2dOutputShape(
        intArrayOf(1, shape[0], shape[1], shape[2]),
        scale
    )
    return intArrayOf(out[1], out[2], out[3])
}

/**
 * Per-sample shape produced by Conv1d given the layer's kernel/stride/padding/dilation.
 * Returns `null` when the input shape is unknown or not rank-2 `(C_in, L)`.
 */
internal fun Conv1dImpl<*, *>.nextShapeFor(currentShape: IntArray?): IntArray? {
    val shape = currentShape ?: return null
    if (shape.size != 2) return null
    val out = ConvShapeUtils.conv1dOutputShape(
        intArrayOf(1, shape[0], shape[1]),
        intArrayOf(outChannels, inChannels / groups, kernelSize),
        stride, padding, dilation
    )
    return intArrayOf(out[1], out[2])
}

/**
 * Per-sample shape produced by Conv2d given the layer's kernel/stride/padding/dilation.
 * Returns `null` when the input shape is unknown or not rank-3 `(C_in, H, W)`.
 */
internal fun Conv2dImpl<*, *>.nextShapeFor(currentShape: IntArray?): IntArray? {
    val shape = currentShape ?: return null
    if (shape.size != 3) return null
    val out = ConvShapeUtils.conv2dOutputShape(
        intArrayOf(1, shape[0], shape[1], shape[2]),
        intArrayOf(outChannels, inChannels / groups, kernelSize.first, kernelSize.second),
        stride, padding, dilation
    )
    return intArrayOf(out[1], out[2], out[3])
}

/**
 * Per-sample shape produced by Conv3d given the layer's kernel/stride/padding/dilation.
 * Returns `null` when the input shape is unknown or not rank-4 `(C_in, D, H, W)`.
 */
internal fun Conv3dImpl<*, *>.nextShapeFor(currentShape: IntArray?): IntArray? {
    val shape = currentShape ?: return null
    if (shape.size != 4) return null
    val out = ConvShapeUtils.conv3dOutputShape(
        intArrayOf(1, shape[0], shape[1], shape[2], shape[3]),
        intArrayOf(outChannels, inChannels / groups, kernelSize.first, kernelSize.second, kernelSize.third),
        stride, padding, dilation
    )
    return intArrayOf(out[1], out[2], out[3], out[4])
}

internal fun flattenedDimensionFor(currentShape: IntArray, startDim: Int, endDim: Int): Int? {
    val tensorRank = currentShape.size + 1
    val sd = if (startDim < 0) tensorRank + startDim else startDim
    val ed = if (endDim < 0) tensorRank + endDim else endDim
    require(sd in 0 until tensorRank && ed in 0 until tensorRank && sd <= ed) {
        "Invalid flatten range startDim=$startDim, endDim=$endDim for tensor rank $tensorRank"
    }
    val perSampleStart = (sd - 1).coerceAtLeast(0)
    val perSampleEndInclusive = ed - 1
    if (perSampleEndInclusive < 0 || perSampleStart > perSampleEndInclusive) return null
    var product = 1
    for (i in perSampleStart..perSampleEndInclusive) product *= currentShape[i]
    return product
}

public class StageImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    private val id: String,
    public val kClass: KClass<T>
) : NeuralNetworkDsl<T, V> {
    public val modules: MutableList<Module<T, V>> = mutableListOf<Module<T, V>>()
    public var lastDimension: Int = 0
    public var inputDimension: Int = 0
    /** Per-sample shape (batch excluded) tracked through spatial layers; `null` if unknown. */
    public var currentShape: IntArray? = null

    public fun create(): Module<T, V> = MLP(*modules.toTypedArray(), name = id)

    override fun input(inputSize: Int, id: String, requiresGrad: Boolean) {
        lastDimension = inputSize
        currentShape = intArrayOf(inputSize)
        modules.add(Input(name = getDefaultName(id, "Input", modules.size), requiresGrad = requiresGrad))
    }

    override fun input(inputShape: IntArray, id: String, requiresGrad: Boolean) {
        require(inputShape.isNotEmpty()) { "input(inputShape) requires at least one dimension" }
        currentShape = inputShape.copyOf()
        lastDimension = inputShape.fold(1) { a, b -> a * b }
        modules.add(Input(name = getDefaultName(id, "Input", modules.size), requiresGrad = requiresGrad))
    }

    override fun flatten(id: String, content: FLATTEN<T, V>.() -> Unit) {
        val impl = FlattenImpl<T, V>(
            executionContext,
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
        val shape = currentShape
        val inferred = if (shape != null) flattenedDimensionFor(shape, impl.startDim, impl.endDim) else null
        if (inferred != null) {
            lastDimension = inferred
            currentShape = intArrayOf(inferred)
        }
        // If we couldn't infer (no input shape declared), leave lastDimension untouched.
        // A subsequent dense() will surface the missing dimension as a clear error.
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE<T, V>.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        currentShape = intArrayOf(outputDimension)
        val impl = DenseImpl<T, V>(
            executionContext,
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size),
            kClass = kClass,
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE<T, V>.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl<T, V>(
            executionContext,
            inputDimension = lastDimension,
            _outputDimension = 0, // Will be set in content block via units property
            id = getDefaultName(id, "linear", modules.size),
            kClass = kClass
        )
        impl.content()
        // Update lastDimension based on the units set in the content block
        lastDimension = impl.outputDimension
        currentShape = intArrayOf(impl.outputDimension)
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun gru(hiddenSize: Int, id: String, content: GRU<T, V>.() -> Unit) {
        val inputSize = lastDimension
        lastDimension = hiddenSize
        currentShape = intArrayOf(hiddenSize)
        val impl = GruImpl<T, V>(
            executionContext,
            inputSize = inputSize,
            hiddenSize = hiddenSize,
            id = getDefaultName(id, "gru", modules.size),
            kClass = kClass,
        )
        impl.content()
        modules += impl.create()
    }

    override fun activation(id: String, activation: (Tensor<T, V>) -> Tensor<T, V>) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl<T, V>(executionContext, kClass)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.currentShape = currentShape?.copyOf()
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        currentShape = sequentialImpl.currentShape?.copyOf()
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(executionContext, id, kClass)
        stageImpl.lastDimension = lastDimension
        stageImpl.currentShape = currentShape?.copyOf()
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        currentShape = stageImpl.currentShape?.copyOf()
        modules += stageImpl.create()
    }

    override fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TLayer : DType> dense(
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision stage that handles conversion
        TODO("Mixed-precision stage implementation needed")
    }

    override fun batchNorm(
        numFeatures: Int,
        eps: Double,
        momentum: Double,
        affine: Boolean,
        id: String
    ) {
        modules.add(
            BatchNormalization(
                numFeatures = numFeatures,
                eps = eps,
                momentum = momentum,
                affine = affine,
                name = getDefaultName(id, "BatchNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun groupNorm(
        numGroups: Int,
        numChannels: Int,
        eps: Double,
        affine: Boolean,
        id: String
    ) {
        modules.add(
            GroupNormalization(
                numGroups = numGroups,
                numChannels = numChannels,
                eps = eps,
                affine = affine,
                name = getDefaultName(id, "GroupNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun layerNorm(
        normalizedShape: IntArray,
        eps: Double,
        elementwiseAffine: Boolean,
        id: String
    ) {
        modules.add(
            LayerNormalization(
                normalizedShape = normalizedShape,
                eps = eps,
                elementwiseAffine = elementwiseAffine,
                name = getDefaultName(id, "LayerNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun conv2d(
        outChannels: Int,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV2D<T, V>.() -> Unit
    ) {
        // Create Conv2dImpl with default inChannels=1, can be modified via DSL
        val conv2dImpl = Conv2dImpl<T, V>(
            executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 3 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv2d", modules.size),
            kClass = kClass
        )

        // Apply the content block to configure the layer
        conv2dImpl.content()

        // Create and add the Conv2d module
        modules.add(conv2dImpl.create())
        currentShape = conv2dImpl.nextShapeFor(currentShape)
    }

    override fun conv2d(
        id: String,
        content: CONV2D<T, V>.() -> Unit
    ) {
        val conv2dImpl = Conv2dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 3 }?.get(0) ?: 1,
            initialOutChannels = 1,
            initialKernelSize = 1 to 1,
            initialStride = 1 to 1,
            initialPadding = 0 to 0,
            initialDilation = 1 to 1,
            initialGroups = 1,
            initialBias = true,
            id = getDefaultName(id, "Conv2d", modules.size),
            kClass = kClass
        )
        conv2dImpl.content()
        modules.add(conv2dImpl.create())
        currentShape = conv2dImpl.nextShapeFor(currentShape)
    }


    override fun maxPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        id: String
    ) {
        modules += MaxPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            name = getDefaultName(id, "MaxPool2d", modules.size)
        )
        currentShape = pool2dNextShape(currentShape, kernelSize, stride, padding)
    }

    override fun maxPool2d(
        id: String,
        content: MAXPOOL2D<T, V>.() -> Unit
    ) {
        val impl = MaxPool2dImpl<T, V>(
            executionContext = executionContext,
            initialKernelSize = 1 to 1,
            initialStride = 1 to 1,
            initialPadding = 0 to 0,
            id = getDefaultName(id, "MaxPool2d", modules.size)
        )
        impl.content()
        modules += impl.create()
        currentShape = pool2dNextShape(currentShape, impl.kernelSize, impl.stride, impl.padding)
    }

    override fun upsample2d(
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean,
        id: String
    ) {
        modules += Upsample2d(
            scale = scale,
            mode = mode,
            alignCorners = alignCorners,
            name = getDefaultName(id, "Upsample2d", modules.size)
        )
        currentShape = upsample2dNextShape(currentShape, scale)
    }

    override fun upsample2d(id: String, content: UPSAMPLE2D<T, V>.() -> Unit) {
        val impl = Upsample2dImpl<T, V>(
            executionContext = executionContext,
            initialScale = 2 to 2,
            initialMode = UpsampleMode.Nearest,
            initialAlignCorners = false,
            id = getDefaultName(id, "Upsample2d", modules.size)
        )
        impl.content()
        modules += impl.create()
        currentShape = upsample2dNextShape(currentShape, impl.scale)
    }

    override fun conv1d(
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV1D<T, V>.() -> Unit
    ) {
        val conv1dImpl = Conv1dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 2 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv1d", modules.size),
            kClass = kClass
        )
        conv1dImpl.content()
        modules.add(conv1dImpl.create())
        currentShape = conv1dImpl.nextShapeFor(currentShape)
    }

    override fun conv3d(
        outChannels: Int,
        kernelSize: Triple<Int, Int, Int>,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV3D<T, V>.() -> Unit
    ) {
        val conv3dImpl = Conv3dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 4 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv3d", modules.size),
            kClass = kClass
        )
        conv3dImpl.content()
        modules.add(conv3dImpl.create())
        currentShape = conv3dImpl.nextShapeFor(currentShape)
    }

    override fun avgPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean,
        id: String
    ) {
        modules += AvgPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            countIncludePad = countIncludePad,
            name = getDefaultName(id, "AvgPool2d", modules.size)
        )
        currentShape = pool2dNextShape(currentShape, kernelSize, stride, padding)
    }

    override fun softmax(dim: Int, id: String) {
        modules += Softmax<T, V>(dim, getDefaultName(id, "Softmax", modules.size))
        // Softmax does not change feature dimension
    }

    // LLM / Transformer layer implementations moved to skainet-transformers llm-core TransformerDsl.kt
}

public class NeuralNetworkDslImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    public val kClass: KClass<T>
) : NeuralNetworkDsl<T, V> {

    public val modules: MutableList<Module<T, V>> = mutableListOf<Module<T, V>>()
    public var lastDimension: Int = 0
    /** Per-sample shape (batch excluded) tracked through spatial layers; `null` if unknown. */
    public var currentShape: IntArray? = null

    public fun create(): Module<T, V> = NetworkBuilder<T, V>().add(*modules.toTypedArray()).build()

    override fun input(inputSize: Int, id: String, requiresGrad: Boolean) {
        lastDimension = inputSize
        currentShape = intArrayOf(inputSize)
        modules.add(Input(name = getDefaultName(id, "Input", modules.size), requiresGrad = requiresGrad))
    }

    override fun input(inputShape: IntArray, id: String, requiresGrad: Boolean) {
        require(inputShape.isNotEmpty()) { "input(inputShape) requires at least one dimension" }
        currentShape = inputShape.copyOf()
        lastDimension = inputShape.fold(1) { a, b -> a * b }
        modules.add(Input(name = getDefaultName(id, "Input", modules.size), requiresGrad = requiresGrad))
    }


    override fun flatten(id: String, content: FLATTEN<T, V>.() -> Unit) {
        val impl = FlattenImpl<T, V>(
            executionContext = executionContext,
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
        val shape = currentShape
        val inferred = if (shape != null) flattenedDimensionFor(shape, impl.startDim, impl.endDim) else null
        if (inferred != null) {
            lastDimension = inferred
            currentShape = intArrayOf(inferred)
        }
        // If we couldn't infer (no input shape declared), leave lastDimension untouched.
        // A subsequent dense() will surface the missing dimension as a clear error.
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE<T, V>.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        currentShape = intArrayOf(outputDimension)
        val impl = DenseImpl<T, V>(
            executionContext = executionContext,
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size),
            kClass = kClass
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE<T, V>.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl<T, V>(
            executionContext,
            inputDimension = lastDimension,
            _outputDimension = 0, // Will be set in content block via units property
            id = getDefaultName(id, "linear", modules.size),
            kClass = kClass,
        )
        impl.content()
        // Update lastDimension based on the units set in the content block
        lastDimension = impl.outputDimension
        currentShape = intArrayOf(impl.outputDimension)
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun gru(hiddenSize: Int, id: String, content: GRU<T, V>.() -> Unit) {
        val inputSize = lastDimension
        lastDimension = hiddenSize
        currentShape = intArrayOf(hiddenSize)
        val impl = GruImpl<T, V>(
            executionContext,
            inputSize = inputSize,
            hiddenSize = hiddenSize,
            id = getDefaultName(id, "gru", modules.size),
            kClass = kClass,
        )
        impl.content()
        modules += impl.create()
    }

    override fun activation(id: String, activation: (Tensor<T, V>) -> Tensor<T, V>) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl<T, V>(executionContext, kClass)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.currentShape = currentShape?.copyOf()
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        currentShape = sequentialImpl.currentShape?.copyOf()
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(executionContext, id, kClass)
        stageImpl.lastDimension = lastDimension
        stageImpl.currentShape = currentShape?.copyOf()
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        currentShape = stageImpl.currentShape?.copyOf()
        modules += stageImpl.create()
    }

    override fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TLayer : DType> dense(
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision stage that handles conversion
        TODO("Mixed-precision stage implementation needed")
    }

    override fun batchNorm(
        numFeatures: Int,
        eps: Double,
        momentum: Double,
        affine: Boolean,
        id: String
    ) {
        modules.add(
            BatchNormalization(
                numFeatures = numFeatures,
                eps = eps,
                momentum = momentum,
                affine = affine,
                name = getDefaultName(id, "BatchNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun groupNorm(
        numGroups: Int,
        numChannels: Int,
        eps: Double,
        affine: Boolean,
        id: String
    ) {
        modules.add(
            GroupNormalization(
                numGroups = numGroups,
                numChannels = numChannels,
                eps = eps,
                affine = affine,
                name = getDefaultName(id, "GroupNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun layerNorm(
        normalizedShape: IntArray,
        eps: Double,
        elementwiseAffine: Boolean,
        id: String
    ) {
        modules.add(
            LayerNormalization(
                normalizedShape = normalizedShape,
                eps = eps,
                elementwiseAffine = elementwiseAffine,
                name = getDefaultName(id, "LayerNorm", modules.size),
                dtype = kClass
            )
        )
    }

    override fun conv2d(
        outChannels: Int,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV2D<T, V>.() -> Unit
    ) {
        // Create Conv2dImpl with default inChannels=1, can be modified via DSL
        val conv2dImpl = Conv2dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 3 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv2d", modules.size),
            kClass = kClass
        )

        // Apply the content block to configure the layer
        conv2dImpl.content()

        // Create and add the Conv2d module
        modules.add(conv2dImpl.create())
        currentShape = conv2dImpl.nextShapeFor(currentShape)
    }

    override fun conv2d(
        id: String,
        content: CONV2D<T, V>.() -> Unit
    ) {
        val conv2dImpl = Conv2dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 3 }?.get(0) ?: 1,
            initialOutChannels = 1,
            initialKernelSize = 1 to 1,
            initialStride = 1 to 1,
            initialPadding = 0 to 0,
            initialDilation = 1 to 1,
            initialGroups = 1,
            initialBias = true,
            id = getDefaultName(id, "Conv2d", modules.size),
            kClass = kClass
        )
        conv2dImpl.content()
        modules.add(conv2dImpl.create())
        currentShape = conv2dImpl.nextShapeFor(currentShape)
    }

    override fun maxPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        id: String
    ) {
        modules += MaxPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            name = getDefaultName(id, "MaxPool2d", modules.size)
        )
        currentShape = pool2dNextShape(currentShape, kernelSize, stride, padding)
    }

    override fun maxPool2d(
        id: String,
        content: MAXPOOL2D<T, V>.() -> Unit
    ) {
        val impl = MaxPool2dImpl<T, V>(
            executionContext = executionContext,
            initialKernelSize = 1 to 1,
            initialStride = 1 to 1,
            initialPadding = 0 to 0,
            id = getDefaultName(id, "MaxPool2d", modules.size)
        )
        impl.content()
        modules.add(impl.create())
        currentShape = pool2dNextShape(currentShape, impl.kernelSize, impl.stride, impl.padding)
    }

    override fun upsample2d(
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean,
        id: String
    ) {
        modules += Upsample2d(
            scale = scale,
            mode = mode,
            alignCorners = alignCorners,
            name = getDefaultName(id, "Upsample2d", modules.size)
        )
        currentShape = upsample2dNextShape(currentShape, scale)
    }

    override fun upsample2d(
        id: String,
        content: UPSAMPLE2D<T, V>.() -> Unit
    ) {
        val impl = Upsample2dImpl<T, V>(
            executionContext = executionContext,
            initialScale = 2 to 2,
            initialMode = UpsampleMode.Nearest,
            initialAlignCorners = false,
            id = getDefaultName(id, "Upsample2d", modules.size)
        )
        impl.content()
        modules.add(impl.create())
        currentShape = upsample2dNextShape(currentShape, impl.scale)
    }

    override fun conv1d(
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV1D<T, V>.() -> Unit
    ) {
        val conv1dImpl = Conv1dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 2 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv1d", modules.size),
            kClass = kClass
        )
        conv1dImpl.content()
        modules.add(conv1dImpl.create())
        currentShape = conv1dImpl.nextShapeFor(currentShape)
    }

    override fun conv3d(
        outChannels: Int,
        kernelSize: Triple<Int, Int, Int>,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int,
        bias: Boolean,
        id: String,
        content: CONV3D<T, V>.() -> Unit
    ) {
        val conv3dImpl = Conv3dImpl<T, V>(
            executionContext = executionContext,
            initialInChannels = currentShape?.takeIf { it.size == 4 }?.get(0) ?: 1,
            initialOutChannels = outChannels,
            initialKernelSize = kernelSize,
            initialStride = stride,
            initialPadding = padding,
            initialDilation = dilation,
            initialGroups = groups,
            initialBias = bias,
            id = getDefaultName(id, "Conv3d", modules.size),
            kClass = kClass
        )
        conv3dImpl.content()
        modules.add(conv3dImpl.create())
        currentShape = conv3dImpl.nextShapeFor(currentShape)
    }

    override fun avgPool2d(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean,
        id: String
    ) {
        modules += AvgPool2d(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            countIncludePad = countIncludePad,
            name = getDefaultName(id, "AvgPool2d", modules.size)
        )
        currentShape = pool2dNextShape(currentShape, kernelSize, stride, padding)
    }

    override fun softmax(dim: Int, id: String) {
        modules += Softmax<T, V>(dim, getDefaultName(id, "Softmax", modules.size))
        // Softmax does not change feature dimension
    }

    // LLM / Transformer layer implementations moved to skainet-transformers llm-core TransformerDsl.kt
}


@NetworkDsl
public class NetworkBuilder<T : DType, V> {
    private val modules = mutableListOf<Module<T, V>>()

    public fun add(vararg modules: Module<T, V>): NetworkBuilder<T, V> {
        this.modules += modules.toList()
        return this
    }

    public fun build(): Module<T, V> = MLP(*modules.toTypedArray(), name = "MLP")
}
