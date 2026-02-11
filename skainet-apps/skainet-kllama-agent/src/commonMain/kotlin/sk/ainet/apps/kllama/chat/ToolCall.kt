package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonObject

/**
 * Represents a tool call emitted by the model.
 *
 * @param id Unique identifier for this tool call (used to correlate with TOOL responses).
 * @param name Name of the tool to invoke.
 * @param arguments Parsed JSON arguments for the tool.
 */
public data class ToolCall(
    val id: String,
    val name: String,
    val arguments: JsonObject
)
