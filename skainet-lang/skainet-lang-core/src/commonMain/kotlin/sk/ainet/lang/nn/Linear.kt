package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.*
import sk.ainet.lang.types.DType
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.nn.topology.biasOrNull
import sk.ainet.lang.nn.topology.weights

/**
 * Linear layer (a.k.a. fully connected dense layer). This layer applies a linear transformation to the input data.
 * The weights and biases are learned during training.
 *
 * Passing `null` for [initBias] creates a bias-less projection (`y = x W^T`),
 * the equivalent of PyTorch's `nn.Linear(..., bias=False)` — used for example
 * by GPT-style Q/K/V projections (`qkv_bias=False`) and weight-tied output
 * heads. A bias-less layer registers only the weight parameter, so parameter
 * counts and checkpoints match architectures defined without bias.
 *
 * The class is `open` so adapter-style layers (e.g. LoRA) can subclass it and
 * augment [onForward] or [params] while reusing the base projection.
 *
 * @param inFeatures Number of input features
 * @param outFeatures Number of output features
 * @param name Name of the module
 * @param initWeights Initial weights
 * @param initBias Initial bias, or `null` for a layer without bias
 */

public open class Linear<T : DType, V> @kotlin.jvm.JvmOverloads constructor(
    inFeatures: Int,
    outFeatures: Int,
    override val name: String = "Linear",
    initWeights: Tensor<T, V>,
    initBias: Tensor<T, V>? = null,
    public val trainable: Boolean = true
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        // Validate weights shape: expected [outFeatures, inFeatures]
        val wShape = initWeights.shape.dimensions
        require(initWeights.rank == 2 && wShape[0] == outFeatures && wShape[1] == inFeatures) {
            "Linear($name): initWeights shape must be [outFeatures, inFeatures]=[${outFeatures}, ${inFeatures}], but was ${initWeights.shape}"
        }
        // Validate bias shape (when present): allow [outFeatures] or [1, outFeatures]
        if (initBias != null) {
            val bShape = initBias.shape.dimensions
            val biasOk = when (initBias.rank) {
                1 -> bShape[0] == outFeatures
                2 -> bShape[0] == 1 && bShape[1] == outFeatures
                else -> false
            }
            require(biasOk) {
                "Linear($name): initBias shape must be [outFeatures] or [1, outFeatures] with outFeatures=${outFeatures}, but was ${initBias.shape}"
            }
        }
    }

    override val params: List<ModuleParameter<T, V>> = buildList {
        add(ModuleParameter.WeightParameter("$name.weight", initWeights, trainable))
        if (initBias != null) {
            add(ModuleParameter.BiasParameter("$name.bias", initBias, trainable))
        }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    @Suppress("UNCHECKED_CAST")
    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val weight = params.weights().value
        val bias = params.biasOrNull()?.value

        val weightTransposed = weight.t()
        val matmulResult = input.matmul(weightTransposed)
        if (bias == null) return matmulResult

        // If input is a 1D vector, ensure bias is also 1D to avoid broadcasting to [1, out]
        val result = if (input.rank == 1 && bias.rank == 2 && bias.shape.dimensions[0] == 1) {
            val outFeatures = bias.shape.dimensions[1]
            val bias1d = bias.reshape(Shape(outFeatures))
            matmulResult + bias1d
        } else {
            matmulResult + bias
        }
        return result as Tensor<T, V>
    }
}
