package sk.ainet.apps.bert.cli

import sk.ainet.models.bert.BertRuntime
import sk.ainet.models.bert.HuggingFaceTokenizer
import sk.ainet.models.bert.loadBertWeights
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime

private fun usage(): Nothing {
    println("Usage: kbert-demo <model-dir> [--iterations N]")
    println()
    println("  <model-dir>    Directory containing model.safetensors, vocab.txt, config.json")
    println("  --iterations N Number of embeddings for throughput test (default: 100)")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.isEmpty()) usage()

        val modelDir = Path.of(args[0])
        if (!modelDir.exists()) error("Model directory not found: $modelDir")

        var iterations = 100
        var i = 1
        while (i < args.size) {
            when (args[i]) {
                "--iterations" -> {
                    iterations = args.getOrNull(i + 1)?.toIntOrNull()
                        ?: error("--iterations requires an integer argument")
                    i += 2
                }
                else -> error("Unknown flag: ${args[i]}")
            }
        }

        // Detect config
        val config = detectConfig(modelDir)

        println("=== SKaiNET LEAF IR Demo ===")
        println("Model: ${modelDir.fileName}")
        println("Config: hidden=${config.hiddenSize}, layers=${config.numHiddenLayers}, heads=${config.numAttentionHeads}, projection=${config.projectionDim ?: "none"}")
        println()

        // Load tokenizer
        val vocabPath = modelDir.resolve("vocab.txt")
        if (!vocabPath.exists()) error("vocab.txt not found in $modelDir")
        print("Loading tokenizer... ")
        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabPath.readText())
        println("done (vocab=${tokenizer.vocabSize})")

        // Load model weights using multi-loader to pick up both model.safetensors and 2_Dense/model.safetensors
        print("Loading model weights... ")
        val ctx = DirectCpuExecutionContext()
        val loaders = buildList {
            val mainFile = resolveModelFile(modelDir)
            add(SafeTensorsParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(mainFile.toString()) },
                onProgress = { _, _, _ -> }
            ))
            val denseFile = modelDir.resolve("2_Dense/model.safetensors")
            if (denseFile.exists()) {
                add(SafeTensorsParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(denseFile.toString()) },
                    onProgress = { _, _, _ -> }
                ))
            }
        }
        val weights = loadBertWeights(loaders, ctx, FP32::class, config)
        println("done (${loaders.size} file(s))")

        val runtime = BertRuntime(ctx, weights, FP32::class)
        val expectedDim = config.projectionDim ?: config.hiddenSize

        // Helper: encode text and return float list
        fun encode(text: String): List<Float> {
            val tok = tokenizer.encodeWithMetadata(text)
            return runtime.encode(tok.inputIds, tok.attentionMask, tok.tokenTypeIds).expectFloatBuffer()
        }

        // ── Section 1: Single Embedding Sanity Check ──
        println()
        println("=== 1. Single Embedding Sanity Check ===")
        val query1 = "What is artificial intelligence?"
        val emb1 = encode(query1)
        println("Query: \"$query1\"")
        println("Embedding dim: ${emb1.size} (expected: $expectedDim)")
        if (emb1.size != expectedDim) {
            println("FAIL: dimension mismatch!")
            exitProcess(1)
        }
        val nonZero = emb1.count { it != 0f }
        val allFinite = emb1.all { it.isFinite() }
        val min = emb1.min()
        val max = emb1.max()
        val mean = emb1.map { it.toDouble() }.average()
        println(String.format(Locale.ROOT, "Non-zero: %d/%d, All finite: %s", nonZero, emb1.size, allFinite))
        println(String.format(Locale.ROOT, "Range: [%.6f, %.6f], Mean: %.6f", min, max, mean))
        println("PASS")

        // ── Section 2: Semantic Similarity Check ──
        println()
        println("=== 2. Semantic Similarity Check ===")
        val query2 = "Define artificial intelligence"
        val query3 = "What is the weather today?"
        val emb2 = encode(query2)
        val emb3 = encode(query3)
        val sim12 = cosineSimilarity(emb1, emb2)
        val sim13 = cosineSimilarity(emb1, emb3)
        println(String.format(Locale.ROOT, "sim(\"%s\", \"%s\") = %.4f", query1, query2, sim12))
        println(String.format(Locale.ROOT, "sim(\"%s\", \"%s\") = %.4f", query1, query3, sim13))
        if (sim12 > sim13) {
            println("PASS: related pair scores higher than unrelated pair")
        } else {
            println("FAIL: expected related pair to score higher")
        }

        // ── Section 3: Small IR Retrieval Test ──
        println()
        println("=== 3. Small IR Retrieval Test ===")
        val baseQuery = "MongoDB is a NoSQL database"
        val candidates = listOf(
            "MongoDB stores data in documents",
            "PostgreSQL is a relational database",
            "The cat sat on the mat",
            "NoSQL databases are non-relational",
            "MongoDB uses BSON format"
        )
        val baseEmb = encode(baseQuery)
        println("Base query: \"$baseQuery\"")
        println()

        val scores = candidates.mapIndexed { idx, doc ->
            val docEmb = encode(doc)
            val sim = cosineSimilarity(baseEmb, docEmb)
            println(String.format(Locale.ROOT, "  [%d] sim=%.4f  %s", idx, sim, doc))
            idx to sim
        }
        val ranked = scores.sortedByDescending { it.second }
        val bestIdx = ranked.first().first
        println()
        println(String.format(Locale.ROOT, "Best match: [%d] score=%.4f", bestIdx, ranked.first().second))
        val mongoRelated = setOf(0, 3, 4)
        if (bestIdx in mongoRelated) {
            println("PASS: best match is MongoDB-related")
        } else {
            println("FAIL: expected a MongoDB-related doc to rank first")
        }

        // ── Section 4: Performance / Throughput Test ──
        println()
        println("=== 4. Performance / Throughput Test ($iterations iterations) ===")
        val elapsed = measureTime {
            for (j in 0 until iterations) {
                encode("$query1 $j")
            }
        }
        val avgMs = elapsed.inWholeNanoseconds / 1_000_000.0 / iterations
        println(String.format(Locale.ROOT, "Total: %s for %d embeddings", elapsed, iterations))
        println(String.format(Locale.ROOT, "Avg: %.3f ms/embedding", avgMs))

        println()
        println("=== Demo Complete ===")
    }
}
