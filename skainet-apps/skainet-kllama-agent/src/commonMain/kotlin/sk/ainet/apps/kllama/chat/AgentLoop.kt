package sk.ainet.apps.kllama.chat

import kotlin.random.Random
import sk.ainet.apps.kllama.agent.GenerateResult
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.lang.types.DType

/**
 * Configuration for the agent loop.
 *
 * @param maxToolRounds Maximum number of tool-call rounds before forcing a final response.
 * @param maxTokensPerRound Maximum tokens to generate in each round.
 * @param temperature Sampling temperature.
 * @param random Random generator for sampling.
 */
public data class AgentConfig @kotlin.jvm.JvmOverloads constructor(
    val maxToolRounds: Int = 5,
    val maxTokensPerRound: Int = 512,
    val temperature: Float = 0.7f,
    val random: Random = Random.Default
)

/**
 * Callback interface for observing the agent loop.
 */
public interface AgentListener {
    /** Called when the model produces text (may be called multiple times per round). */
    public fun onToken(token: String) {}

    /** Called when the model's full response for a round is available. */
    public fun onAssistantMessage(text: String) {}

    /** Called when tool calls are detected in the model's response. */
    public fun onToolCalls(calls: List<ToolCall>) {}

    /** Called when a tool returns a result. */
    public fun onToolResult(call: ToolCall, result: String) {}

    /** Called when the agent loop finishes. */
    public fun onComplete(finalResponse: String) {}
}

/**
 * Orchestrates the generate -> parse -> execute -> re-prompt cycle.
 *
 * The agent loop:
 * 1. Formats the conversation using the [template]
 * 2. Generates tokens until EOS or max tokens
 * 3. Parses the output for tool calls
 * 4. If tool calls are found, executes them and appends results to the conversation
 * 5. Repeats until no tool calls are found or [AgentConfig.maxToolRounds] is reached
 *
 * @param T The DType of the model weights.
 * @param runtime The inference runtime for token generation.
 * @param template The chat template for formatting prompts.
 * @param toolRegistry Registry of available tools.
 * @param eosTokenId The EOS token ID used to detect generation completion.
 * @param config Agent configuration.
 * @param decode Function to decode a token ID to a string fragment.
 */
public class AgentLoop<T : DType>(
    private val runtime: InferenceRuntime<T>,
    private val template: ChatTemplate,
    private val toolRegistry: ToolRegistry,
    private val eosTokenId: Int,
    private val config: AgentConfig = AgentConfig(),
    private val decode: (Int) -> String
) {

    /**
     * Run the agent loop with the given conversation history.
     *
     * @param messages Initial conversation messages (typically system + user).
     * @param listener Optional listener for observing the loop.
     * @return The final assistant response text.
     */
    public fun run(
        messages: MutableList<ChatMessage>,
        listener: AgentListener? = null
    ): String {
        val tools = toolRegistry.definitions()
        var lastResponse = ""

        for (round in 0 until config.maxToolRounds) {
            // Reset runtime state for each round (re-process full context)
            runtime.reset()

            // Format the full conversation into a prompt
            val prompt = template.apply(messages, tools, addGenerationPrompt = true)
            val promptTokens = encodePrompt(prompt)

            // Generate until EOS or max tokens
            val result: GenerateResult = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = config.maxTokensPerRound,
                eosTokenId = eosTokenId,
                temperature = config.temperature,
                random = config.random,
                onToken = { tokenId -> listener?.onToken(decode(tokenId)) },
                decode = decode
            )

            lastResponse = result.text
            listener?.onAssistantMessage(lastResponse)

            // Parse for tool calls
            val toolCalls = ToolCallParser.parse(lastResponse)

            if (toolCalls.isEmpty()) {
                // No tool calls — this is the final response
                messages.add(ChatMessage(role = ChatRole.ASSISTANT, content = lastResponse))
                listener?.onComplete(lastResponse)
                return lastResponse
            }

            // Tool calls found — execute them
            listener?.onToolCalls(toolCalls)
            messages.add(
                ChatMessage(
                    role = ChatRole.ASSISTANT,
                    content = lastResponse,
                    toolCalls = toolCalls
                )
            )

            for (call in toolCalls) {
                val toolResult = toolRegistry.execute(call)
                listener?.onToolResult(call, toolResult)
                messages.add(
                    ChatMessage(
                        role = ChatRole.TOOL,
                        content = toolResult,
                        toolCallId = call.id
                    )
                )
            }
        }

        // Max rounds exhausted — return last response
        listener?.onComplete(lastResponse)
        return lastResponse
    }

    /**
     * Simple prompt-to-tokens encoding.
     * Encodes each character individually — the runtime's tokenizer should be used
     * for real encoding. This is a fallback for when only a decode function is available.
     */
    private fun encodePrompt(prompt: String): IntArray {
        // This is handled at the CLI level where we have access to the tokenizer.
        // The AgentLoop receives pre-encoded tokens in practice.
        // For now, we store the prompt and let the CLI layer handle encoding.
        throw UnsupportedOperationException(
            "AgentLoop.encodePrompt should not be called directly. " +
                "Use AgentLoop.run() with pre-encoded prompts via runWithEncoder()."
        )
    }

    /**
     * Run the agent loop with an explicit tokenizer encode function.
     *
     * @param messages Conversation history.
     * @param encode Function to encode a string to token IDs.
     * @param listener Optional listener.
     * @return The final assistant response text.
     */
    public fun runWithEncoder(
        messages: MutableList<ChatMessage>,
        encode: (String) -> IntArray,
        listener: AgentListener? = null
    ): String {
        val tools = toolRegistry.definitions()
        var lastResponse = ""

        for (round in 0 until config.maxToolRounds) {
            runtime.reset()

            val prompt = template.apply(messages, tools, addGenerationPrompt = true)
            val promptTokens = encode(prompt)

            val result: GenerateResult = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = config.maxTokensPerRound,
                eosTokenId = eosTokenId,
                temperature = config.temperature,
                random = config.random,
                onToken = { tokenId -> listener?.onToken(decode(tokenId)) },
                decode = decode
            )

            lastResponse = result.text
            listener?.onAssistantMessage(lastResponse)

            val toolCalls = ToolCallParser.parse(lastResponse)

            if (toolCalls.isEmpty()) {
                messages.add(ChatMessage(role = ChatRole.ASSISTANT, content = lastResponse))
                listener?.onComplete(lastResponse)
                return lastResponse
            }

            listener?.onToolCalls(toolCalls)
            messages.add(
                ChatMessage(
                    role = ChatRole.ASSISTANT,
                    content = lastResponse,
                    toolCalls = toolCalls
                )
            )

            for (call in toolCalls) {
                val toolResult = toolRegistry.execute(call)
                listener?.onToolResult(call, toolResult)
                messages.add(
                    ChatMessage(
                        role = ChatRole.TOOL,
                        content = toolResult,
                        toolCallId = call.id
                    )
                )
            }
        }

        listener?.onComplete(lastResponse)
        return lastResponse
    }
}
