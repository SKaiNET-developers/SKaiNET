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
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import sk.ainet.io.gguf.llama.loadLlamaRuntimeWeights
import sk.ai.net.sample.llama2.model.TokenizerUtils

private val scope = MainScope()

fun main() {
    scope.launch {
        val output = document.getElementById("output") ?: return@launch
        val runButton = document.getElementById("run")

        suspend fun runDemo() {
            output.textContent = "Loading model...\n"
            try {
                val modelPath = "models/model.gguf" // change to your filename if different
                val format = if (modelPath.endsWith(".gguf")) LlamaWeightLoader.Format.GGUF else LlamaWeightLoader.Format.KARPATHY_BIN

                val runtime = loadRuntime(modelPath, format)
                val tokenizer = loadTokenizer("models/tokenizer.bin", runtime.weights.metadata.vocabSize)

                output.appendChild(document.createTextNode("Generating...\n"))
                val promptTokens = tokenizer.encode("Hello")
                runtime.reset()
                runtime.generate(prompt = promptTokens.toIntArray(), steps = 64, temperature = 0.8f) { id ->
                    output.appendChild(document.createTextNode(tokenizer.decode(id)))
                }
            } catch (t: Throwable) {
                console.error("Failed to run LLaMA demo", t)
                output.appendChild(document.createTextNode("\nError: ${t.message}"))
            }
        }

        // Run once on load
        runDemo()
        // Allow reruns
        runButton?.addEventListener("click", { scope.launch { runDemo() } })
    }
}

private suspend fun loadRuntime(path: String, format: LlamaWeightLoader.Format): LlamaRuntime {
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
    // Disambiguate buffered() by casting Buffer to RawSource explicitly
    val buffer = Buffer().apply { write(bytes) }
    val source: Source = (buffer as RawSource).buffered()
    val ctx = DirectCpuExecutionContext()
    val weights = loadLlamaRuntimeWeights(
        ctx = ctx,
        sourceProvider = { source },
        format = format,
        quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32
    )
    return LlamaRuntime(ctx, weights)
}

private suspend fun loadTokenizer(path: String, vocabSize: Int): sk.ai.net.sample.llama2.model.Tokenizer {
    val resp: Response = (window.fetch(path) as Promise<Response>).await()
    if (!resp.ok) error("Failed to fetch tokenizer: ${resp.statusText}")
    val buf: ArrayBuffer = (resp.arrayBuffer() as Promise<ArrayBuffer>).await()
    val view = DataView(buf)
    val length = view.byteLength
    val bytes = ByteArray(length)
    for (i in 0 until length) {
        bytes[i] = view.getUint8(i).toByte()
    }
    val buffer = Buffer().apply { write(bytes) }
    val source: Source = (buffer as RawSource).buffered()
    return TokenizerUtils.buildTokenizer(source, vocabSize)
}
