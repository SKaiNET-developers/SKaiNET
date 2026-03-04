package sk.ainet.apps.bert.cli

import sk.ainet.models.bert.BertIngestion
import sk.ainet.models.bert.BertModelConfig
import sk.ainet.models.bert.BertRuntime
import sk.ainet.models.bert.HuggingFaceTokenizer
import sk.ainet.models.bert.MDBR_LEAF_IR_CONFIG
import sk.ainet.models.bert.BertRuntimeWeights
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime

private fun usage(): Nothing {
    println("Usage: kbert <model-dir> \"query text\" [\"doc text\"]")
    println()
    println("  <model-dir>   Directory containing model.safetensors, vocab.txt, config.json")
    println("  \"query text\"  Text to encode")
    println("  \"doc text\"    Optional second text — if given, prints cosine similarity")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.isEmpty()) usage()

        val modelDir = Path.of(args[0])
        if (!modelDir.exists()) error("Model directory not found: $modelDir")

        val textA = args.getOrNull(1) ?: usage()
        val textB = args.getOrNull(2)

        // Resolve model files
        val safetensorsPath = resolveModelFile(modelDir)
        val vocabPath = modelDir.resolve("vocab.txt")
        if (!vocabPath.exists()) error("vocab.txt not found in $modelDir")

        // Detect config
        val config = detectConfig(modelDir)

        println("Model: ${modelDir.fileName}")
        println("Config: hidden=${config.hiddenSize}, layers=${config.numHiddenLayers}, heads=${config.numAttentionHeads}")
        println()

        // Load tokenizer
        print("Loading tokenizer... ")
        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabPath.readText())
        println("done (vocab=${tokenizer.vocabSize})")

        // Load model weights
        print("Loading model weights... ")
        val ctx = DirectCpuExecutionContext()
        val ingestion = BertIngestion<FP32>(ctx, FP32::class, config)
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(safetensorsPath.toString()) },
            onProgress = { _, _, _ -> }
        )
        val weights: BertRuntimeWeights<FP32>
        val loadElapsed = measureTime { weights = ingestion.load(loader) }
        println("done (${loadElapsed})")

        val runtime = BertRuntime(ctx, weights, FP32::class)

        // Encode text(s)
        val tokOutputA = tokenizer.encodeWithMetadata(textA)
        println("\nEncoding: \"$textA\" (${tokOutputA.inputIds.size} tokens)")
        val embA: Tensor<FP32, Float>
        val encodeElapsed = measureTime {
            embA = runtime.encode(tokOutputA.inputIds, tokOutputA.attentionMask, tokOutputA.tokenTypeIds)
        }
        println("Encoded in $encodeElapsed")
        val vecA = embA.expectFloatBuffer()
        println("Embedding (first 8): ${vecA.take(8).joinToString(", ") { "%.6f".format(it) }}")
        println("Embedding dim: ${vecA.size}")

        if (textB != null) {
            val tokOutputB = tokenizer.encodeWithMetadata(textB)
            println("\nEncoding: \"$textB\" (${tokOutputB.inputIds.size} tokens)")
            val embB = runtime.encode(tokOutputB.inputIds, tokOutputB.attentionMask, tokOutputB.tokenTypeIds)
            val vecB = embB.expectFloatBuffer()
            println("Embedding (first 8): ${vecB.take(8).joinToString(", ") { "%.6f".format(it) }}")

            // Cosine similarity
            val sim = cosineSimilarity(vecA, vecB)
            println("\nCosine similarity: %.6f".format(sim))
        }
    }
}

internal fun resolveModelFile(modelDir: Path): Path {
    val candidates = listOf("model.safetensors", "pytorch_model.safetensors")
    for (name in candidates) {
        val p = modelDir.resolve(name)
        if (p.exists()) return p
    }
    // Try any .safetensors file
    val dir = modelDir.toFile()
    val found = dir.listFiles()?.firstOrNull { it.extension == "safetensors" }
    if (found != null) return found.toPath()
    error("No .safetensors file found in $modelDir")
}

internal fun detectConfig(modelDir: Path): BertModelConfig {
    val configPath = modelDir.resolve("config.json")
    if (!configPath.exists()) {
        println("No config.json found, using MDBR_LEAF_IR_CONFIG defaults")
        return MDBR_LEAF_IR_CONFIG
    }
    val json = configPath.readText()
    // Check 2_Dense/config.json for projection dim (sentence-transformers layout)
    val denseConfigPath = modelDir.resolve("2_Dense/config.json")
    val denseJson = if (denseConfigPath.exists()) denseConfigPath.readText() else null
    return parseConfigJson(json, denseJson)
}

internal fun parseConfigJson(json: String, denseJson: String? = null): BertModelConfig {
    fun extractInt(source: String, key: String, default: Int): Int {
        val pattern = Regex("\"$key\"\\s*:\\s*(\\d+)")
        return pattern.find(source)?.groupValues?.get(1)?.toIntOrNull() ?: default
    }
    fun extractDouble(source: String, key: String, default: Double): Double {
        val pattern = Regex("\"$key\"\\s*:\\s*([\\d.eE\\-+]+)")
        return pattern.find(source)?.groupValues?.get(1)?.toDoubleOrNull() ?: default
    }

    // Projection dim comes from 2_Dense/config.json (sentence-transformers layout)
    val projDim = if (denseJson != null) {
        extractInt(denseJson, "out_features", 0).let { if (it > 0) it else null }
    } else {
        null
    }

    return BertModelConfig(
        vocabSize = extractInt(json, "vocab_size", 30522),
        hiddenSize = extractInt(json, "hidden_size", 384),
        numHiddenLayers = extractInt(json, "num_hidden_layers", 6),
        numAttentionHeads = extractInt(json, "num_attention_heads", 12),
        intermediateSize = extractInt(json, "intermediate_size", 1536),
        maxPositionEmbeddings = extractInt(json, "max_position_embeddings", 512),
        typeVocabSize = extractInt(json, "type_vocab_size", 2),
        layerNormEps = extractDouble(json, "layer_norm_eps", 1e-12),
        projectionDim = projDim
    )
}

internal fun cosineSimilarity(a: List<Float>, b: List<Float>): Float {
    require(a.size == b.size) { "Vectors must have same dimension" }
    var dot = 0f
    var normA = 0f
    var normB = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
    return if (denom > 0f) dot / denom else 0f
}

internal fun <T : DType> Tensor<T, Float>.expectFloatBuffer(): List<Float> {
    val data = this.data
    if (data is FloatArrayTensorData<*>) return data.buffer.toList()
    return data.copyToFloatArray().toList()
}
