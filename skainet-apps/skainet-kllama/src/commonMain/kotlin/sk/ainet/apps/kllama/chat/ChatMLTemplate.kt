package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * ChatML template used by Hermes, OpenHermes, and other ChatML-based models.
 *
 * Format:
 * ```
 * <|im_start|>system
 * {content}<|im_end|>
 * <|im_start|>user
 * {content}<|im_end|>
 * <|im_start|>assistant
 * ```
 */
public class ChatMLTemplate : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()

        // If tools are provided, inject them into a system message
        if (tools.isNotEmpty()) {
            sb.append("<|im_start|>system\n")
            sb.append("You are a helpful assistant with access to the following tools:\n\n")
            for (tool in tools) {
                val toolJson = buildJsonObject {
                    put("name", tool.name)
                    put("description", tool.description)
                    put("parameters", tool.parameters)
                }
                sb.append(Json.encodeToString(JsonObject.serializer(), toolJson))
                sb.append("\n\n")
            }
            sb.append("To call a tool, respond with:\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {...}}\n</tool_call>")
            sb.append("<|im_end|>\n")
        }

        for (msg in messages) {
            val role = when (msg.role) {
                ChatRole.TOOL -> "tool"
                else -> msg.role.roleName
            }
            sb.append("<|im_start|>").append(role).append("\n")
            sb.append(msg.content)
            sb.append("<|im_end|>\n")
        }

        if (addGenerationPrompt) {
            sb.append("<|im_start|>assistant\n")
        }

        return sb.toString()
    }
}
