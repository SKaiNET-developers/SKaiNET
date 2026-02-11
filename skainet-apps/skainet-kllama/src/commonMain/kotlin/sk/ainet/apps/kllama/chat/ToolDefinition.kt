package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.JsonObject

/**
 * Definition of a tool that can be called by the model.
 *
 * @param name Unique name of the tool (e.g. "calculator", "web_search").
 * @param description Human-readable description of what the tool does.
 * @param parameters JSON Schema object describing the tool's parameters.
 */
public data class ToolDefinition(
    val name: String,
    val description: String,
    val parameters: JsonObject
)
