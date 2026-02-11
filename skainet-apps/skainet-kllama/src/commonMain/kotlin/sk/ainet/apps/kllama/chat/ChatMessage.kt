package sk.ainet.apps.kllama.chat

/**
 * Roles in a chat conversation following the standard LLM chat convention.
 */
public enum class ChatRole {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL;

    /** Lowercase role name used in chat templates. */
    public val roleName: String get() = name.lowercase()
}

/**
 * A single message in a chat conversation.
 *
 * @param role The role of the message sender.
 * @param content The text content of the message.
 * @param toolCalls Optional list of tool calls made by the assistant.
 * @param toolCallId Optional ID linking a TOOL response to its originating tool call.
 */
public data class ChatMessage(
    val role: ChatRole,
    val content: String,
    val toolCalls: List<ToolCall>? = null,
    val toolCallId: String? = null
)
