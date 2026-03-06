package sk.ainet.apps.kllama.java

import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.lang.types.FP32
import sk.ainet.apps.llm.Tokenizer
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.function.Consumer

/**
 * Java-friendly session for LLaMA text generation.
 *
 * Wraps the internal LlamaRuntime and Tokenizer, providing a simple
 * generate API that handles encoding/decoding and state management.
 *
 * Implements AutoCloseable for try-with-resources usage.
 *
 * Example usage from Java:
 * ```java
 * try (KLlamaSession session = KLlamaJava.loadGGUF(Path.of("model.gguf"))) {
 *     String result = session.generate("Hello, world!");
 *     System.out.println(result);
 *
 *     // Streaming
 *     session.generate("Tell me a story", config, token -> System.out.print(token));
 * }
 * ```
 */
public class KLlamaSession(
    internal val runtime: LlamaRuntime<FP32>,
    internal val tokenizer: Tokenizer,
    private val eosTokenId: Int,
    public val systemPrompt: String? = null,
    private val closeAction: Runnable? = null
) : AutoCloseable {

    /**
     * Generate text from the given prompt.
     *
     * @param prompt The input prompt.
     * @param config Generation configuration. Uses defaults if not specified.
     * @return The generated text.
     */
    @JvmOverloads
    public fun generate(prompt: String, config: GenerationConfig = GenerationConfig.defaults()): String {
        runtime.reset()
        val fullPrompt = if (systemPrompt != null) "$systemPrompt\n\n$prompt" else prompt
        val tokens = tokenizer.encode(fullPrompt)
        val result = runtime.generateUntilStop(
            prompt = tokens,
            maxTokens = config.maxTokens,
            eosTokenId = eosTokenId,
            temperature = config.temperature,
            decode = { tokenizer.decode(it) }
        )
        return result.text
    }

    /**
     * Generate text with streaming token output.
     *
     * @param prompt The input prompt.
     * @param config Generation configuration.
     * @param tokenConsumer Called for each generated token text fragment.
     * @return The complete generated text.
     */
    public fun generate(prompt: String, config: GenerationConfig, tokenConsumer: Consumer<String>): String {
        runtime.reset()
        val fullPrompt = if (systemPrompt != null) "$systemPrompt\n\n$prompt" else prompt
        val tokens = tokenizer.encode(fullPrompt)
        val result = runtime.generateUntilStop(
            prompt = tokens,
            maxTokens = config.maxTokens,
            eosTokenId = eosTokenId,
            temperature = config.temperature,
            onToken = { tokenId -> tokenConsumer.accept(tokenizer.decode(tokenId)) },
            decode = { tokenizer.decode(it) }
        )
        return result.text
    }

    /**
     * Generate text asynchronously using virtual threads.
     *
     * @param prompt The input prompt.
     * @param config Generation configuration.
     * @return A CompletableFuture that completes with the generated text.
     */
    @JvmOverloads
    public fun generateAsync(
        prompt: String,
        config: GenerationConfig = GenerationConfig.defaults()
    ): CompletableFuture<String> {
        return CompletableFuture.supplyAsync(
            { generate(prompt, config) },
            Executors.newVirtualThreadPerTaskExecutor()
        )
    }

    /**
     * Release all resources (Arena, MemorySegment, etc.).
     */
    override fun close() {
        closeAction?.run()
    }
}
