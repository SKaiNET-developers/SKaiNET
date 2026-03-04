package sk.ainet.apps.kllama.browser

import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.await
import kotlinx.coroutines.launch
import kotlinx.io.Source
import kotlinx.io.buffered
import kotlinx.io.Buffer
import kotlinx.io.RawSource
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.DataView
import org.w3c.fetch.Response
import kotlin.js.Promise
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeInterface
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.llama.loadLlamaRuntimeWeights
import sk.ainet.apps.llm.Tokenizer

private val scope = MainScope()

fun main() {
    scope.launch {
        val output = document.getElementById("output") ?: return@launch
        val runButton = document.getElementById("run")

        suspend fun runDemo() {
            output.textContent = "Loading model...\n"
            try {
                val modelPath = "models/model.gguf"

                val (runtime, tokenizer) = loadRuntimeAndTokenizer(modelPath)

                output.appendChild(document.createTextNode("Generating...\n"))
                val promptTokens = tokenizer.encode("Hello")
                runtime.reset()
                runtime.generate(prompt = promptTokens, steps = 64, temperature = 0.8f) { id ->
                    output.appendChild(document.createTextNode(tokenizer.decode(id)))
                }
            } catch (t: Throwable) {
                println("Failed to run LLaMA demo: ${t.message}")
                output.appendChild(document.createTextNode("\nError: ${t.message}"))
            }
        }

        // Run once on load
        runDemo()
        // Allow reruns
        runButton?.addEventListener("click", { scope.launch { runDemo() } })
    }
}

@Suppress("UNCHECKED_CAST")
private suspend fun loadRuntimeAndTokenizer(path: String): Pair<LlamaRuntimeInterface<*>, Tokenizer> {
    val resp: Response = (window.fetch(path) as Promise<Response>).await()
    if (!resp.ok) error("Failed to fetch model: ${resp.statusText}")
    // On Wasm, use arrayBuffer() and feed bytes into a kotlinx-io Buffer as Source
    val buf: ArrayBuffer = (resp.arrayBuffer() as Promise<ArrayBuffer>).await()
    val view = DataView(buf)
    val length = view.byteLength
    val bytes = ByteArray(length)
    for (i in 0 until length) {
        bytes[i] = view.getUint8(i).toByte()
    }

    // Create source for loading weights
    val buffer1 = Buffer().apply { write(bytes) }
    val source1: Source = (buffer1 as RawSource).buffered()
    val ctx = DirectCpuExecutionContext()
    val weights = loadLlamaRuntimeWeights(
        ctx = ctx,
        sourceProvider = { source1 },
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )

    // Create source for loading tokenizer (need fresh buffer as source is consumed)
    val buffer2 = Buffer().apply { write(bytes) }
    val source2: Source = (buffer2 as RawSource).buffered()
    val tokenizer = GGUFTokenizer.fromSource(source2)

    val backend = sk.ainet.apps.kllama.CpuAttentionBackend(ctx, weights, sk.ainet.lang.types.FP32::class)
    return LlamaRuntime(ctx, weights, backend, sk.ainet.lang.types.FP32::class) to tokenizer
}
