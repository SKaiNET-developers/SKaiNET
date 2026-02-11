package sk.ainet.apps.kllama

import kotlin.math.exp
import kotlin.random.Random
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Result of EOS-aware generation.
 *
 * @param tokens The generated token IDs (excluding the prompt).
 * @param text The decoded text (if a tokenizer was provided).
 * @param stoppedByEos True if generation stopped because EOS was emitted.
 */
public data class GenerateResult(
    val tokens: List<Int>,
    val text: String,
    val stoppedByEos: Boolean
)

/**
 * Generate tokens until an EOS token is produced or [maxTokens] is reached.
 *
 * Unlike [LlamaRuntimeInterface.generate], this function:
 * - Stops when the model emits [eosTokenId]
 * - Does NOT prepend BOS automatically (the caller is responsible for encoding the
 *   full prompt including special tokens via the chat template)
 * - Returns a [GenerateResult] with all generated tokens and decoded text
 *
 * @param prompt Encoded prompt token IDs (should include BOS if needed).
 * @param maxTokens Maximum number of tokens to generate.
 * @param eosTokenId The EOS token ID to stop on.
 * @param temperature Sampling temperature (0 = greedy).
 * @param random Random generator for sampling.
 * @param onToken Optional callback invoked for each generated token.
 * @param decode Optional function to decode a token ID to a string.
 */
public fun <T : DType> LlamaRuntimeInterface<T>.generateUntilStop(
    prompt: IntArray,
    maxTokens: Int,
    eosTokenId: Int,
    temperature: Float = 0.8f,
    random: Random = Random.Default,
    onToken: ((Int) -> Unit)? = null,
    decode: ((Int) -> String)? = null
): GenerateResult {
    // Feed prompt tokens through the model
    var lastLogits: Tensor<T, Float>? = null
    for (tokenId in prompt) {
        lastLogits = forward(tokenId)
    }

    if (lastLogits == null) {
        return GenerateResult(emptyList(), "", false)
    }

    val generated = mutableListOf<Int>()
    val textBuilder = StringBuilder()
    var stoppedByEos = false

    var logits: Tensor<T, Float> = lastLogits
    for (step in 0 until maxTokens) {
        val nextToken = sampleFromLogits<T>(logits, temperature, random)

        if (nextToken == eosTokenId) {
            stoppedByEos = true
            break
        }

        generated.add(nextToken)
        onToken?.invoke(nextToken)
        decode?.let { textBuilder.append(it(nextToken)) }

        logits = forward(nextToken)
    }

    return GenerateResult(generated, textBuilder.toString(), stoppedByEos)
}

/**
 * Sample a token ID from a logits tensor.
 *
 * @param logits The logits tensor (1D, vocabSize).
 * @param temperature Sampling temperature. Values <= 1e-6 use greedy (argmax).
 * @param random Random generator.
 * @return The sampled token ID.
 */
public fun <T : DType> sampleFromLogits(
    logits: Tensor<T, Float>,
    temperature: Float,
    random: Random = Random.Default
): Int {
    val buf = logits.toFloatArray()

    // Greedy (argmax) for near-zero temperature
    if (temperature <= 1e-6f) {
        var best = 0
        var bestVal = buf[0]
        for (i in 1 until buf.size) {
            if (buf[i] > bestVal) {
                bestVal = buf[i]
                best = i
            }
        }
        return best
    }

    // Temperature-scaled softmax sampling
    var maxLogit = Float.NEGATIVE_INFINITY
    for (i in buf.indices) {
        val v = buf[i] / temperature
        buf[i] = v
        if (v > maxLogit) maxLogit = v
    }
    var sum = 0f
    for (i in buf.indices) {
        val e = exp((buf[i] - maxLogit).toDouble()).toFloat()
        buf[i] = e
        sum += e
    }
    val r = random.nextFloat() * sum
    var acc = 0f
    for (i in buf.indices) {
        acc += buf[i]
        if (acc >= r) return i
    }
    return buf.lastIndex
}

/**
 * Extract a FloatArray from a tensor, using the fast path if available.
 */
private fun <T : DType> Tensor<T, Float>.toFloatArray(): FloatArray {
    val data = this.data
    if (data is FloatArrayTensorData<*>) return data.buffer.copyOf()
    return data.copyToFloatArray()
}
