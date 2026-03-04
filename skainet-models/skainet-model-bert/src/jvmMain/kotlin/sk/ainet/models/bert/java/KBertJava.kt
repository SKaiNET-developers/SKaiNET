@file:JvmName("KBertJava")

package sk.ainet.models.bert.java

import kotlinx.coroutines.runBlocking
import sk.ainet.models.bert.*
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.readText

/**
 * Java-friendly facade for loading and running BERT models.
 *
 * Example usage from Java:
 * ```java
 * try (KBertSession session = KBertJava.loadSafeTensors(Path.of("model-dir"))) {
 *     float[] embedding = session.encode("Hello world");
 *     float similarity = session.similarity("query", "document");
 * }
 * ```
 */
public object KBertJava {

    /**
     * Load a BERT model from a HuggingFace directory containing
     * model.safetensors, vocab.txt, and optionally config.json.
     *
     * @param modelDir Path to the model directory.
     * @return A KBertSession that implements AutoCloseable.
     */
    @JvmStatic
    public fun loadSafeTensors(modelDir: Path): KBertSession {
        require(modelDir.exists()) { "Model directory not found: $modelDir" }

        val safetensorsPath = modelDir.resolve("model.safetensors")
        require(safetensorsPath.exists()) { "model.safetensors not found in $modelDir" }

        val vocabPath = modelDir.resolve("vocab.txt")
        require(vocabPath.exists()) { "vocab.txt not found in $modelDir" }

        // Detect config from config.json or use defaults
        val config = detectConfig(modelDir)

        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabPath.readText())
        val ctx = DirectCpuExecutionContext()

        val ingestion = BertIngestion<FP32>(ctx, FP32::class, config)
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(safetensorsPath.toString()) },
            onProgress = { _, _, _ -> }
        )

        val weights = runBlocking { ingestion.load(loader) }
        val runtime = BertRuntime(ctx, weights, FP32::class)

        return KBertSession(runtime, tokenizer)
    }

    private fun detectConfig(modelDir: Path): BertModelConfig {
        val configFile = modelDir.resolve("config.json")
        if (!configFile.exists()) return MDBR_LEAF_IR_CONFIG

        val json = configFile.readText()
        fun extractInt(key: String, default: Int): Int {
            val pattern = Regex("\"$key\"\\s*:\\s*(\\d+)")
            return pattern.find(json)?.groupValues?.get(1)?.toIntOrNull() ?: default
        }
        fun extractDouble(key: String, default: Double): Double {
            val pattern = Regex("\"$key\"\\s*:\\s*([\\d.eE\\-+]+)")
            return pattern.find(json)?.groupValues?.get(1)?.toDoubleOrNull() ?: default
        }

        return BertModelConfig(
            vocabSize = extractInt("vocab_size", 30522),
            hiddenSize = extractInt("hidden_size", 768),
            numHiddenLayers = extractInt("num_hidden_layers", 12),
            numAttentionHeads = extractInt("num_attention_heads", 12),
            intermediateSize = extractInt("intermediate_size", 3072),
            maxPositionEmbeddings = extractInt("max_position_embeddings", 512),
            typeVocabSize = extractInt("type_vocab_size", 2),
            layerNormEps = extractDouble("layer_norm_eps", 1e-12)
        )
    }
}

/**
 * Java-friendly session for BERT encoding and similarity.
 *
 * Implements AutoCloseable for try-with-resources usage.
 */
public class KBertSession(
    private val runtime: BertRuntime<FP32>,
    private val tokenizer: HuggingFaceTokenizer
) : AutoCloseable {

    /**
     * Encode text into an embedding vector.
     *
     * @param text The input text.
     * @return A float array representing the embedding.
     */
    public fun encode(text: String): FloatArray {
        val tokOutput = tokenizer.encodeWithMetadata(text)
        val embedding = runtime.encode(tokOutput.inputIds, tokOutput.attentionMask, tokOutput.tokenTypeIds)
        return embedding.data.copyToFloatArray()
    }

    /**
     * Compute cosine similarity between two texts.
     *
     * @param textA First text.
     * @param textB Second text.
     * @return Cosine similarity in [-1, 1].
     */
    public fun similarity(textA: String, textB: String): Float {
        val embA = encode(textA)
        val embB = encode(textB)
        return cosineSimilarity(embA, embB)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Embedding dimensions must match" }
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = Math.sqrt(normA.toDouble()) * Math.sqrt(normB.toDouble())
        return if (denom < 1e-12) 0f else (dot / denom).toFloat()
    }

    override fun close() {
        // No-op for now — CPU context doesn't need explicit cleanup
    }
}
