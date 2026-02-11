package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Chat template for Llama 3 / 3.1 / 3.2 family models.
 *
 * Format:
 * ```
 * <|begin_of_text|><|start_header_id|>system<|end_header_id|>
 *
 * {content}<|eot_id|>
 * <|start_header_id|>user<|end_header_id|>
 *
 * {content}<|eot_id|>
 * <|start_header_id|>assistant<|end_header_id|>
 *
 * ```
 */
public class Llama3ChatTemplate : ChatTemplate {

    override fun apply(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        addGenerationPrompt: Boolean
    ): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")

        // If tools are provided, inject them into the system prompt
        if (tools.isNotEmpty()) {
            sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
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
            sb.append("To call a tool, respond with a JSON object: {\"name\": \"tool_name\", \"arguments\": {...}}")
            sb.append("<|eot_id|>")
        }

        for (msg in messages) {
            val role = when (msg.role) {
                ChatRole.TOOL -> "tool"
                else -> msg.role.roleName
            }
            sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n")
            sb.append(msg.content)
            sb.append("<|eot_id|>")
        }

        if (addGenerationPrompt) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }

        return sb.toString()
    }
}
