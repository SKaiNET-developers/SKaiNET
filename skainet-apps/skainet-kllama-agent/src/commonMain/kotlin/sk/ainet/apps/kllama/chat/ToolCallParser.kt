package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Parses tool calls from model output text.
 *
 * Supports two formats:
 * 1. **Hermes/ChatML style**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
 * 2. **Llama 3.1 style**: Bare JSON `{"name": "...", "arguments": {...}}`
 */
public object ToolCallParser {

    private val hermesPattern = Regex("""<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>""")

    private val json = Json { ignoreUnknownKeys = true }

    /**
     * Attempt to parse one or more tool calls from [text].
     *
     * @return List of parsed [ToolCall]s (empty if no tool calls were found).
     */
    public fun parse(text: String): List<ToolCall> {
        // Try Hermes-style <tool_call> tags first
        val hermesMatches = buildList {
            var match = hermesPattern.find(text)
            while (match != null) {
                add(match)
                match = hermesPattern.find(text, match.range.last + 1)
            }
        }
        if (hermesMatches.isNotEmpty()) {
            return hermesMatches.mapNotNull { match ->
                parseJsonToolCall(match.groupValues[1])
            }
        }

        // Try bare JSON object with "name" key (Llama 3.1 style)
        val trimmed = text.trim()
        if (trimmed.startsWith("{") && trimmed.contains("\"name\"")) {
            val parsed = parseJsonToolCall(trimmed)
            if (parsed != null) return listOf(parsed)
        }

        return emptyList()
    }

    /**
     * Check whether [text] appears to contain a tool call.
     */
    public fun containsToolCall(text: String): Boolean {
        if (text.contains("<tool_call>")) return true
        val trimmed = text.trim()
        return trimmed.startsWith("{") && trimmed.contains("\"name\"") && trimmed.contains("\"arguments\"")
    }

    private fun parseJsonToolCall(jsonStr: String): ToolCall? {
        return try {
            val obj = json.parseToJsonElement(jsonStr).jsonObject
            val name = obj["name"]?.jsonPrimitive?.content ?: return null
            val arguments = obj["arguments"]?.jsonObject ?: JsonObject(emptyMap())
            val id = obj["id"]?.jsonPrimitive?.content ?: generateCallId()
            ToolCall(id = id, name = name, arguments = arguments)
        } catch (_: Exception) {
            null
        }
    }

    private var callCounter = 0

    private fun generateCallId(): String {
        return "call_${callCounter++}"
    }
}
