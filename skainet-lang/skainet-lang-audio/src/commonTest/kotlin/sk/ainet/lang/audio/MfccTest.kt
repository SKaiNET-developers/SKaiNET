package sk.ainet.lang.audio

import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MfccTest {
    private val ctx = DefaultNeuralNetworkExecutionContext()

    @Test
    fun mfcc_shapes_with_energy_and_deltas() {
        val sampleRate = 16_000
        val samples = FloatArray(640) { idx ->
            sin(2.0 * PI * 440 * idx / sampleRate).toFloat()
        }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(1, samples.size), FP32::class, samples)
        val config = MfccConfig(
            sampleRate = sampleRate,
            frameSize = 160,
            hopSize = 80,
            fftSize = 256,
            melBands = 20,
            coeffs = 8,
            includeEnergy = true,
            deltas = true,
            deltaWindow = 2
        )
        val plan = AudioPlan.build(config)
        val output = mfcc(ctx, input, config, plan)

        val expectedFrames = 1 + (samples.size - config.frameSize) / config.hopSize
        val expectedCoeffCount = (config.coeffs + 1) * 3

        assertEquals(1, output.shape[0])
        assertEquals(expectedFrames, output.shape[1])
        assertEquals(expectedCoeffCount, output.shape[2])

        val flattened = output.expectBuffer()
        assertTrue(flattened.any { it.isFinite() })
        val firstStatic = flattened.first()
        val lastStatic = flattened[config.coeffs]
        assertTrue(abs(firstStatic) >= abs(lastStatic))
    }

    @Test
    fun mfcc_handles_batchless_input() {
        val sampleRate = 8_000
        val samples = FloatArray(320) { idx ->
            sin(2.0 * PI * 220 * idx / sampleRate).toFloat()
        }
        val input = ctx.fromFloatArray<FP32, Float>(Shape(samples.size), FP32::class, samples)
        val config = MfccConfig(
            sampleRate = sampleRate,
            frameSize = 80,
            hopSize = 40,
            fftSize = 128,
            melBands = 16,
            coeffs = 6,
            includeEnergy = false,
            deltas = false
        )
        val output = mfcc(ctx, input, config)

        val expectedFrames = 1 + (samples.size - config.frameSize) / config.hopSize
        assertEquals(expectedFrames, output.shape[1])
        assertEquals(config.coeffs, output.shape[2])
        assertTrue(output.expectBuffer().any { it != 0f })
    }

    private fun Tensor<FP32, Float>.expectBuffer(): FloatArray {
        val data = this.data
        require(data is FloatArrayTensorData<*>) { "Expected dense float data" }
        return data.buffer
    }
}
