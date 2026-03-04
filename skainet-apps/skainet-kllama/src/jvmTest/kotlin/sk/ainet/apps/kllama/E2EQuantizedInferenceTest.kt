package sk.ainet.apps.kllama

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.models.llama.loadLlamaRuntimeWeightsStreaming
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking

/**
 * End-to-end integration test: loads a real Q8_0 GGUF model via NATIVE_OPTIMIZED,
 * converts to MemorySegment-backed tensors, runs inference, and verifies output.
 *
 * Requires TinyLlama Q8_0 GGUF at the path below. Skips gracefully if not found.
 */
class E2EQuantizedInferenceTest {

    companion object {
        private val MODEL_PATH = Path.of(
            System.getProperty("user.home"),
            ".lmstudio/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        )
    }

    private fun skipIfNoModel(): Boolean {
        if (!MODEL_PATH.exists()) {
            println("SKIPPING: Model not found at $MODEL_PATH")
            return true
        }
        return false
    }

    @Test
    fun `NATIVE_OPTIMIZED loads and converts Q8 tensors to MemorySegment`() {
        if (skipIfNoModel()) return
        runBlocking {
            val arena = Arena.ofConfined()
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                    allowQuantized = true
                )
            )

            val rawWeights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }

            println("Loaded model: ${rawWeights.metadata.architecture}")
            println("  Layers: ${rawWeights.metadata.blockCount}, dim: ${rawWeights.metadata.embeddingLength}")
            println("  Quant types: ${rawWeights.quantTypes.size} tensors")
            assertTrue(rawWeights.quantTypes.isNotEmpty(), "Should have quantized tensors")

            val converted = MemSegWeightConverter.convert(rawWeights, ctx, arena)

            // Verify Q8 conversion happened
            val layer0 = converted.layers[0]
            assertTrue(layer0.wq.data is Q8MemorySegmentMarker,
                "wq should be Q8 MemorySegment, got ${layer0.wq.data::class.simpleName}")
            assertTrue(layer0.wk.data is Q8MemorySegmentMarker,
                "wk should be Q8 MemorySegment")
            assertTrue(layer0.ffnGate.data is Q8MemorySegmentMarker,
                "ffnGate should be Q8 MemorySegment")

            println("All ${converted.layers.size} layers converted to Q8 MemorySegment successfully")

            arena.close()
            memSegFactory.close()
        }
    }

    @Test
    fun `end-to-end inference produces non-empty output`() {
        if (skipIfNoModel()) return
        runBlocking {
            val arena = Arena.ofShared()
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                    allowQuantized = true
                )
            )

            val rawWeights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val weights = MemSegWeightConverter.convert(rawWeights, ctx, arena)

            val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
            val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)

            // Load tokenizer from GGUF
            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val prompt = "Hello"
            val promptTokens = tokenizer.encode(prompt)
            assertTrue(promptTokens.isNotEmpty(), "Prompt should encode to tokens")

            val steps = 16
            val generated = mutableListOf<String>()

            runtime.generate(prompt = promptTokens, steps = steps, temperature = 0.0f) { id ->
                generated.add(tokenizer.decode(id))
            }

            val output = generated.joinToString("")
            println("Prompt: '$prompt'")
            println("Generated ($steps tokens): '$output'")
            assertTrue(generated.isNotEmpty(), "Should generate at least one token")
            assertTrue(generated.size == steps, "Should generate exactly $steps tokens")

            arena.close()
            memSegFactory.close()
        }
    }

    @Test
    fun `benchmark NATIVE_OPTIMIZED MemorySegment inference`() {
        if (skipIfNoModel()) return
        runBlocking {
            val arena = Arena.ofShared()
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                    allowQuantized = true
                )
            )

            val loadStart = System.nanoTime()
            val raw = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val weights = MemSegWeightConverter.convert(raw, ctx, arena)
            val loadTimeMs = (System.nanoTime() - loadStart) / 1_000_000

            val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
            val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)

            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 32

            // Warmup
            runtime.generate(prompt = promptTokens, steps = 4, temperature = 0.0f) { _ -> }

            // Benchmark
            val generated = mutableListOf<String>()
            val inferenceTime = measureTime {
                runtime.generate(prompt = promptTokens, steps = steps, temperature = 0.0f) { id ->
                    generated.add(tokenizer.decode(id))
                }
            }.inWholeMilliseconds

            val tokPerSec = steps.toDouble() / inferenceTime * 1000
            val output = generated.joinToString("")

            println("=== NATIVE_OPTIMIZED + MemorySegment Benchmark ===")
            println("Model: TinyLlama 1.1B Q8_0")
            println("Load time: ${loadTimeMs}ms")
            println("Prompt: '$prompt'")
            println("Output: '$output'")
            println("Steps: $steps")
            println("Inference time: ${inferenceTime}ms")
            println("Throughput: ${"%.2f".format(tokPerSec)} tok/s")
            println("=================================================")

            assertTrue(tokPerSec > 0, "Should produce positive throughput")

            arena.close()
            memSegFactory.close()
        }
    }

    @Test
    fun `benchmark DEQUANTIZE_TO_FP32 baseline inference`() {
        if (skipIfNoModel()) return
        runBlocking {
            // Use MemorySegment factory so dequantized FP32 weights live off-heap
            val arena = Arena.ofShared()
            val memSegFactory = MemorySegmentTensorDataFactory()
            val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
                config = LlamaLoadConfig(
                    quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                    allowQuantized = true
                )
            )

            val loadStart = System.nanoTime()
            val weights = ingestion.loadStreaming {
                JvmRandomAccessSource.open(MODEL_PATH.toString())
            }
            val loadTimeMs = (System.nanoTime() - loadStart) / 1_000_000

            val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
            val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)

            val tokenizer = JvmRandomAccessSource.open(MODEL_PATH.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }

            val prompt = "The capital of France is"
            val promptTokens = tokenizer.encode(prompt)
            val steps = 32

            // Warmup
            runtime.generate(prompt = promptTokens, steps = 4, temperature = 0.0f) { _ -> }

            // Benchmark
            val generated = mutableListOf<String>()
            val inferenceTime = measureTime {
                runtime.generate(prompt = promptTokens, steps = steps, temperature = 0.0f) { id ->
                    generated.add(tokenizer.decode(id))
                }
            }.inWholeMilliseconds

            val tokPerSec = steps.toDouble() / inferenceTime * 1000
            val output = generated.joinToString("")

            println("=== DEQUANTIZE_TO_FP32 Baseline Benchmark ===")
            println("Model: TinyLlama 1.1B Q8_0 (dequantized to FP32)")
            println("Load time: ${loadTimeMs}ms")
            println("Prompt: '$prompt'")
            println("Output: '$output'")
            println("Steps: $steps")
            println("Inference time: ${inferenceTime}ms")
            println("Throughput: ${"%.2f".format(tokPerSec)} tok/s")
            println("==============================================")

            assertTrue(tokPerSec > 0, "Should produce positive throughput")

            arena.close()
            memSegFactory.close()
        }
    }
}
