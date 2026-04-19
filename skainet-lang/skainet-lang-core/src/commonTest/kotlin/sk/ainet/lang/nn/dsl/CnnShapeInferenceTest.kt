package sk.ainet.lang.nn.dsl

import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Verifies that the DSL tracks per-sample shapes through conv/pool/upsample/flatten
 * so that a downstream `dense()` receives the correct input dimension.
 *
 * Before #535 was fixed, `flatten()` fell back to a hardcoded `1568` (the MNIST CNN
 * value), which broke any other architecture.
 */
class CnnShapeInferenceTest {

    @Test
    fun input_intArray_sets_currentShape_and_flat_lastDimension() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.input(intArrayOf(1, 28, 28))
        assertContentEquals(intArrayOf(1, 28, 28), builder.currentShape)
        assertEquals(1 * 28 * 28, builder.lastDimension)
    }

    @Test
    fun mnist_cnn_chain_infers_1568_after_flatten() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(1, 28, 28))
            // 28 + 2*2 - (5-1) - 1 + 1 = 28
            conv2d(outChannels = 16, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
            // (28 + 0 - 2)/2 + 1 = 14
            maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
            conv2d(outChannels = 32, kernelSize = 5 to 5, stride = 1 to 1, padding = 2 to 2)
            // 14 -> 7
            maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
        }
        assertContentEquals(intArrayOf(32, 7, 7), builder.currentShape)
        builder.flatten()
        assertEquals(32 * 7 * 7, builder.lastDimension)
    }

    @Test
    fun custom_64_channel_cnn_does_not_collide_with_old_1568() {
        // The architecture from issue #535 that used to crash because flatten() hardcoded 1568.
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(3, 32, 32))
            conv2d(outChannels = 32, kernelSize = 3 to 3, padding = 1 to 1)
            // 32 + 0 - (2) ) / 2 + 1 = 16
            maxPool2d(kernelSize = 2 to 2)
            conv2d(outChannels = 64, kernelSize = 3 to 3, padding = 1 to 1)
            maxPool2d(kernelSize = 2 to 2) // 8
        }
        assertContentEquals(intArrayOf(64, 8, 8), builder.currentShape)
        builder.flatten()
        assertEquals(64 * 8 * 8, builder.lastDimension)
    }

    @Test
    fun conv1d_chain_tracks_length_correctly() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(80, 3000))   // (channels, length) — Whisper-style mel input
            conv1d(outChannels = 384, kernelSize = 3, padding = 1)
        }
        // (3000 + 2 - 2 - 1)/1 + 1 = 3000
        assertContentEquals(intArrayOf(384, 3000), builder.currentShape)
    }

    @Test
    fun upsample2d_doubles_spatial_dims() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(8, 16, 16))
            upsample2d(scale = 2 to 2)
        }
        assertContentEquals(intArrayOf(8, 32, 32), builder.currentShape)
    }

    @Test
    fun avgPool2d_reduces_spatial_dims() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(16, 32, 32))
            avgPool2d(kernelSize = 4 to 4, stride = 4 to 4)
        }
        assertContentEquals(intArrayOf(16, 8, 8), builder.currentShape)
    }

    @Test
    fun stage_propagates_shape_in_and_out() {
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.apply {
            input(intArrayOf(1, 28, 28))
            stage("conv1") {
                conv2d(outChannels = 16, kernelSize = 5 to 5, padding = 2 to 2)
                maxPool2d(kernelSize = 2 to 2)
            }
        }
        assertContentEquals(intArrayOf(16, 14, 14), builder.currentShape)
    }

    @Test
    fun flatten_without_input_shape_leaves_lastDimension_untouched() {
        // Backward-compat: building a sequential with bare flatten (no input) must
        // not throw, because some tests use it as a runtime-only module.
        val builder = NeuralNetworkDslImpl<FP32, Float>(
            DefaultNetworkExecutionContext, FP32::class
        )
        builder.flatten()
        assertEquals(0, builder.lastDimension)
        assertNotNull(builder.modules.firstOrNull())
    }
}

private val DefaultNetworkExecutionContext = sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext()
