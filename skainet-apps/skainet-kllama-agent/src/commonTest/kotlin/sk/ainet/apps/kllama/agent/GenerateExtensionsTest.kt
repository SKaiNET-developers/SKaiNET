package sk.ainet.apps.kllama.agent

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

class GenerateExtensionsTest {

    // --- sampleFromLogits tests ---

    @Test
    fun sampleGreedyReturnsArgmax() {
        val logits = floatArrayOf(0.1f, 0.2f, 0.5f, 2.0f, 0.3f)
        val tensor = createFP32Tensor(logits)

        val result = sampleFromLogits(tensor, temperature = 0.0f)
        assertEquals(3, result, "Greedy sampling should return argmax")
    }

    @Test
    fun sampleGreedyWithAllSameValues() {
        val logits = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
        val tensor = createFP32Tensor(logits)

        val result = sampleFromLogits(tensor, temperature = 0.0f)
        assertEquals(0, result, "Greedy with equal values should return first index")
    }

    @Test
    fun sampleWithTemperatureProducesValidIndex() {
        val logits = floatArrayOf(1.0f, 2.0f, 3.0f, 0.5f)
        val tensor = createFP32Tensor(logits)

        val result = sampleFromLogits(tensor, temperature = 1.0f, random = Random(42))
        assertTrue(result in 0..3, "Sampled index should be within range")
    }

    @Test
    fun sampleDistributionFavorsHighLogits() {
        val logits = floatArrayOf(0.0f, 0.0f, 10.0f, 0.0f)

        // With a strong peak at index 2, most samples should be 2
        var count2 = 0
        repeat(100) {
            val t = createFP32Tensor(logits.copyOf())
            if (sampleFromLogits(t, temperature = 0.5f, random = Random(it)) == 2) count2++
        }
        assertTrue(count2 > 90, "Expected most samples at index 2, got $count2/100")
    }

    // --- EOS stop behavior test ---

    @Test
    fun generateUntilStopRespectsEos() {
        val eosTokenId = 2
        var callCount = 0
        val fakeRuntime = object : InferenceRuntime<FP32> {
            override fun reset() { callCount = 0 }
            override fun forward(tokenId: Int): Tensor<FP32, Float> {
                callCount++
                return if (callCount <= 3) {
                    createFP32Tensor(FloatArray(10) { if (it == 5) 10.0f else 0.0f })
                } else {
                    createFP32Tensor(FloatArray(10) { if (it == eosTokenId) 10.0f else 0.0f })
                }
            }
        }

        val result = fakeRuntime.generateUntilStop(
            prompt = intArrayOf(1),
            maxTokens = 100,
            eosTokenId = eosTokenId,
            temperature = 0.0f
        )

        assertEquals(3, result.tokens.size, "Expected 3 tokens before EOS")
        assertTrue(result.tokens.all { it == 5 }, "All generated tokens should be 5")
        assertTrue(result.stoppedByEos, "Should have stopped due to EOS")
    }

    @Test
    fun generateUntilStopRespectsMaxTokens() {
        val fakeRuntime = object : InferenceRuntime<FP32> {
            override fun reset() {}
            override fun forward(tokenId: Int): Tensor<FP32, Float> {
                return createFP32Tensor(FloatArray(10) { if (it == 7) 10.0f else 0.0f })
            }
        }

        val result = fakeRuntime.generateUntilStop(
            prompt = intArrayOf(1),
            maxTokens = 5,
            eosTokenId = 2,
            temperature = 0.0f
        )

        assertEquals(5, result.tokens.size, "Should generate exactly maxTokens")
        assertTrue(!result.stoppedByEos, "Should not have stopped by EOS")
    }

    companion object {
        fun createFP32Tensor(data: FloatArray): Tensor<FP32, Float> {
            val shape = Shape(data.size)
            val tensorData = DenseFloatArrayTensorData<FP32>(shape, data)
            return object : Tensor<FP32, Float> {
                override val data: TensorData<FP32, Float> = tensorData
                override val ops: TensorOps = VoidTensorOps()
                override val dtype: KClass<FP32> = FP32::class
                override val gradState: GradState<FP32, Float> = GradState()
            }
        }
    }
}
