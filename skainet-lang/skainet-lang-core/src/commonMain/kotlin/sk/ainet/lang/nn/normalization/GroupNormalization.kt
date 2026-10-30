package sk.ainet.lang.nn.normalization

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.*
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * GroupNormalization layer - Alternative normalization approach.
 * Normalizes the input by dividing channels into groups and normalizing within each group.
 *
 * @param numGroups Number of groups to divide the channels into
 * @param numChannels Number of channels in the input
 * @param eps Small value added to the denominator for numerical stability
 * @param affine Whether to learn affine parameters (gamma and beta)
 * @param name Name of the module
 * @param initGamma Initial gamma (scale) parameter
 * @param initBeta Initial beta (shift) parameter
 */
public class GroupNormalization<T : DType, V>(
    private val numGroups: Int,
    private val numChannels: Int,
    private val eps: Double = 1e-5,
    private val affine: Boolean = true,
    override val name: String = "GroupNormalization",
    initGamma: Tensor<T, V>? = null,
    initBeta: Tensor<T, V>? = null,
    // Logical element type prescribed by the DSL; keeps placeholder weights typed.
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    init {
        require(numChannels % numGroups == 0) {
            "Number of channels ($numChannels) must be divisible by number of groups ($numGroups)"
        }
    }

    private val channelsPerGroup = numChannels / numGroups

    override val params: List<ModuleParameter<T, V>> = if (affine) {
        listOf(
            ModuleParameter.WeightParameter("$name.weight", initGamma ?: createOnesParameter()),
            ModuleParameter.BiasParameter("$name.bias", initBeta ?: createZerosParameter())
        )
    } else {
        emptyList()
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    @Suppress("UNCHECKED_CAST")
    private fun createOnesParameter(): Tensor<T, V> {
        // Create a placeholder tensor - this is a minimal implementation for tests to pass
        // In a real implementation, this would need proper tensor initialization
        return VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(numChannels)
                override fun get(vararg indices: Int): V = 1.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun createZerosParameter(): Tensor<T, V> {
        // Create a placeholder tensor - this is a minimal implementation for tests to pass
        // In a real implementation, this would need proper tensor initialization
        return VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(numChannels)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
    }

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val inputShape = input.shape
            require(inputShape.rank >= 2) {
                "GroupNormalization expects input of shape (N, C, ...), got rank ${inputShape.rank}"
            }
            require(inputShape.dimensions[1] == numChannels) {
                "Input has ${inputShape.dimensions[1]} channels, expected $numChannels"
            }

            // Compute the numerically-sensitive normalization (mean / variance /
            // sqrt / divide) in f32 regardless of the model dtype, mirroring
            // LayerNormalization: a low-precision variance loses enough precision
            // that `sqrt(var + eps)` can overflow to NaN. The affine (gamma/beta)
            // stays in the model dtype. No-op when the model already runs f32.
            val normalized: Tensor<T, V> = if (input.dtype != FP32::class) {
                val xF32 = input.ops.convert(input, FP32)
                val normF32 = normalizeGroups(xF32)
                val modelDtype: DType = when (input.dtype) {
                    BF16::class -> BF16
                    FP16::class -> FP16
                    else -> FP32
                }
                @Suppress("UNCHECKED_CAST")
                (input.ops.convert(normF32, modelDtype) as Tensor<T, V>)
            } else {
                normalizeGroups(input)
            }

            if (affine) {
                // gamma/beta are per-channel (C); reshape to (1, C, 1, ...) so they
                // broadcast over the batch and spatial dimensions. The reshape must run
                // on the input's ops: the default placeholder params carry VoidTensorOps,
                // whose reshape is shape-only and would zero the values.
                val broadcastShape = Shape(IntArray(inputShape.rank) { idx -> if (idx == 1) numChannels else 1 })
                val gamma = input.ops.reshape(params[0].value, broadcastShape)
                val beta = input.ops.reshape(params[1].value, broadcastShape)
                normalized * gamma + beta
            } else {
                normalized
            }
        }

    /**
     * Normalizes with `torch.nn.GroupNorm` semantics: (N, C, *) is reshaped to
     * (N, G, C/G * prod(*)), each group is normalized with its own biased
     * mean/variance over the last dimension, and the result is reshaped back.
     */
    private fun <D : DType> normalizeGroups(input: Tensor<D, V>): Tensor<D, V> {
        val originalShape = input.shape
        var rest = 1
        for (i in 2 until originalShape.rank) rest *= originalShape.dimensions[i]
        val grouped = input.reshape(Shape(originalShape.dimensions[0], numGroups, channelsPerGroup * rest))

        val mean = grouped.mean(dim = 2).unsqueeze(2) // (N, G, 1)
        val centered = grouped - mean
        // Biased variance (mean of squared deviations), matching torch.nn.GroupNorm.
        val variance = (centered * centered).mean(dim = 2).unsqueeze(2)
        val normalized = centered / (variance + eps).sqrt()

        return normalized.reshape(originalShape)
    }
}