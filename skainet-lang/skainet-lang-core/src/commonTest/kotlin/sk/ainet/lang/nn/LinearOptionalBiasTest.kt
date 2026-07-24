package sk.ainet.lang.nn

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import sk.ainet.lang.nn.topology.bias
import sk.ainet.lang.nn.topology.biasOrNull
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/** Parameter registration for bias-less Linear layers (value behavior is covered in backend-cpu). */
class LinearOptionalBiasTest {

    private val ctx = DefaultNeuralNetworkExecutionContext()

    private fun weights(out: Int, inF: Int) =
        ctx.fromFloatArray<FP32, Float>(Shape(out, inF), FP32::class, FloatArray(out * inF) { 1f })

    @Test
    fun without_bias_only_the_weight_parameter_is_registered() {
        val layer = Linear<FP32, Float>(
            inFeatures = 3, outFeatures = 2, name = "lin",
            initWeights = weights(2, 3), initBias = null,
        )
        assertEquals(1, layer.params.size)
        assertEquals("lin.weight", layer.params.single().name)
        assertNull(layer.params.biasOrNull())
        assertFailsWith<NoSuchElementException> { layer.params.bias() }
    }

    @Test
    fun with_bias_both_parameters_are_registered() {
        val bias = ctx.fromFloatArray<FP32, Float>(Shape(2), FP32::class, floatArrayOf(0f, 0f))
        val layer = Linear<FP32, Float>(
            inFeatures = 3, outFeatures = 2, name = "lin",
            initWeights = weights(2, 3), initBias = bias,
        )
        assertEquals(2, layer.params.size)
        assertEquals("lin.bias", layer.params.bias().name)
    }

    @Test
    fun trainable_parameter_count_reflects_the_missing_bias() {
        val withBias = Linear<FP32, Float>(
            inFeatures = 4, outFeatures = 3, name = "a",
            initWeights = weights(3, 4),
            initBias = ctx.fromFloatArray(Shape(3), FP32::class, FloatArray(3)),
        )
        val withoutBias = Linear<FP32, Float>(
            inFeatures = 4, outFeatures = 3, name = "b",
            initWeights = weights(3, 4), initBias = null,
        )
        assertEquals(15, withBias.trainableParameters().sumOf { it.value.volume })
        assertEquals(12, withoutBias.trainableParameters().sumOf { it.value.volume })
    }
}
