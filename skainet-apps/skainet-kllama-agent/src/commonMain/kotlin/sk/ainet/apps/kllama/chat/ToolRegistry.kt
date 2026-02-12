package sk.ainet.apps.kllama.chat

/**
 * Registry that maps tool names to their [Tool] implementations.
 */
public class ToolRegistry {

    private val tools = mutableMapOf<String, Tool>()

    /** Register a tool. Replaces any existing tool with the same name. */
    public fun register(tool: Tool) {
        tools[tool.definition.name] = tool
    }

    /** Look up a tool by name. */
    public fun get(name: String): Tool? = tools[name]

    /** All registered tool definitions (for injecting into the prompt). */
    public fun definitions(): List<ToolDefinition> = tools.values.map { it.definition }

    /** Execute a [ToolCall] and return the result string. */
    public fun execute(call: ToolCall): String {
        val tool = tools[call.name]
            ?: return "Error: unknown tool '${call.name}'"
        return try {
            tool.execute(call.arguments)
        } catch (e: Exception) {
            "Error executing tool '${call.name}': ${e.message}"
        }
    }
}
