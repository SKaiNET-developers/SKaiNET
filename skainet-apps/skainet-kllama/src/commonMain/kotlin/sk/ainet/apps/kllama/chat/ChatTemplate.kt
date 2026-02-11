package sk.ainet.apps.kllama.chat

/**
 * Formats a list of chat messages (and optional tool definitions) into a single
 * prompt string that the model expects.
 *
 * Different model families use different chat formats (Llama 3, ChatML, etc.).
 */
public interface ChatTemplate {

    /**
     * Apply the template to produce a prompt string.
     *
     * @param messages Conversation history.
     * @param tools Available tool definitions (empty if tool calling is not in use).
     * @param addGenerationPrompt If true, append the assistant turn prefix so the
     *   model knows it should start generating.
     * @return The formatted prompt string.
     */
    public fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition> = emptyList(),
        addGenerationPrompt: Boolean = true
    ): String
}
