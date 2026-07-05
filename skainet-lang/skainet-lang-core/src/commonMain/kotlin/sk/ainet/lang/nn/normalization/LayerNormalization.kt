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
 * LayerNormalization layer - Used in attention mechanisms.
 * Normalizes the input across the last dimension(s).
 *
 * @param normalizedShape The shape of the normalization (typically the last dimension(s))
 * @param eps Small value added to the denominator for numerical stability
 * @param elementwiseAffine Whether to learn elementwise affine parameters (gamma and beta)
 * @param name Name of the module
 * @param initGamma Initial gamma (scale) parameter
 * @param initBeta Initial beta (shift) parameter
 */
public class LayerNormalization<T : DType, V>(
    private val normalizedShape: IntArray,
    private val eps: Double = 1e-5,
    private val elementwiseAffine: Boolean = true,
    override val name: String = "LayerNormalization",
    initGamma: Tensor<T, V>? = null,
    initBeta: Tensor<T, V>? = null,
    // Logical element type prescribed by the DSL; keeps placeholder weights typed.
    private val dtype: KClass<T>? = null,
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = if (elementwiseAffine) {
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
                override val shape = Shape(*normalizedShape)
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
                override val shape = Shape(*normalizedShape)
                override fun get(vararg indices: Int): V = 0.0f as V
                override fun set(vararg indices: Int, value: V) {}
            },
            (dtype ?: Any::class) as KClass<T>
        )
    }

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            // Calculate the dimensions to normalize over (last normalizedShape.size dimensions)
            val inputShape = input.shape.dimensions
            val numNormDims = normalizedShape.size
            val normDims = inputShape.takeLast(numNormDims).toIntArray()
            
            // Verify that the input shape matches the expected normalized shape
            require(normDims.contentEquals(normalizedShape)) {
                "Input shape ${normDims.contentToString()} doesn't match normalized shape ${normalizedShape.contentToString()}"
            }
            
            // Compute the numerically-sensitive normalization (mean / variance /
            // std / divide) in f32, regardless of the model dtype. This is standard
            // LayerNorm practice — PyTorch and JAX upcast fp16/bf16 LayerNorm to f32
            // internally — because a low-precision variance (a sum of many bf16/fp16
            // squares) loses enough precision that `sqrt(var + eps)` can overflow to
            // NaN, and some accelerator backends miscompile the decomposed
            // low-precision reduce/normalize outright. The affine (gamma/beta) stays in
            // the model dtype. No-op when the model already runs f32.
            val normalized: Tensor<T, V> = if (input.dtype != FP32::class) {
                val xF32 = input.ops.convert(input, FP32)
                val mean = calculateLayerMean(xF32)
                val variance = calculateLayerVariance(xF32, mean)
                val normF32 = normalize(xF32, mean, variance)
                val modelDtype: DType = when (input.dtype) {
                    BF16::class -> BF16
                    FP16::class -> FP16
                    else -> FP32
                }
                @Suppress("UNCHECKED_CAST")
                (input.ops.convert(normF32, modelDtype) as Tensor<T, V>)
            } else {
                val mean = calculateLayerMean(input)
                val variance = calculateLayerVariance(input, mean)
                normalize(input, mean, variance)
            }

            if (elementwiseAffine) {
                val gamma = params[0].value // weight parameter
                val beta = params[1].value  // bias parameter
                normalized * gamma + beta
            } else {
                normalized
            }
        }

    private fun <D : DType> calculateLayerMean(input: Tensor<D, V>): Tensor<D, V> {
        var m = input
        for (i in normalizedShape.indices.reversed()) {
            m = m.mean(dim = m.rank - 1)
        }
        // Restore reduced dims so broadcasting works against the original input
        var result = m
        for (i in normalizedShape.indices) {
            result = result.unsqueeze(result.rank)
        }
        return result
    }

    private fun <D : DType> calculateLayerVariance(input: Tensor<D, V>, mean: Tensor<D, V>): Tensor<D, V> {
        val centered = input - mean
        var v = centered * centered
        for (i in normalizedShape.indices.reversed()) {
            v = v.mean(dim = v.rank - 1)
        }
        // Restore reduced dims so broadcasting works against the original input
        var result = v
        for (i in normalizedShape.indices) {
            result = result.unsqueeze(result.rank)
        }
        return result
    }

    private fun <D : DType> normalize(input: Tensor<D, V>, mean: Tensor<D, V>, variance: Tensor<D, V>): Tensor<D, V> {
        return (input - mean) / (variance + eps).sqrt()
    }
}