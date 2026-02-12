package sk.ainet.apps.kllama.chat

/**
 * A tool that can be called by the model during an agent loop.
 *
 * Implementations provide their [definition] (name, description, parameters schema)
 * and an [execute] function that performs the actual work.
 */
public interface Tool {

    /** The tool's definition, used for prompt injection and parsing. */
    public val definition: ToolDefinition

    /**
     * Execute the tool with the given arguments and return a string result.
     *
     * @param arguments The parsed JSON arguments from the model's tool call.
     * @return A string result that will be fed back to the model as a TOOL message.
     */
    public fun execute(arguments: kotlinx.serialization.json.JsonObject): String
}
