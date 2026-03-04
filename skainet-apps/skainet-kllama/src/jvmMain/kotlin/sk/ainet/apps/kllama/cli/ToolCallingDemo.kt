package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.models.llama.LlamaRuntimeInterface
import sk.ainet.apps.kllama.chat.*
import sk.ainet.lang.types.DType
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/**
 * Demo CLI showcasing tool calling with a file-system tool.
 *
 * The model can call `list_files` to list files in a local folder,
 * demonstrating how the agent loop orchestrates real-world tool use.
 *
 * Example interaction:
 * ```
 * User: What files are in /tmp?
 * [Tool Call] list_files({"path": "/tmp"})
 * [Tool Result] list_files -> file1.txt\nfile2.log\n...
 * Assistant: The /tmp directory contains file1.txt, file2.log, ...
 * ```
 */
public class ToolCallingDemo<T : DType>(
    private val runtime: LlamaRuntimeInterface<T>,
    private val tokenizer: GGUFTokenizer,
    private val templateName: String = "llama3"
) {
    private val template: ChatTemplate = when (templateName.lowercase()) {
        "chatml", "hermes" -> ChatMLTemplate()
        else -> Llama3ChatTemplate()
    }

    private val eosTokenId: Int = tokenizer.eosId

    /**
     * Run the tool-calling demo with `list_files` and `calculator` tools.
     */
    public fun run(
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ) {
        val registry = ToolRegistry()
        registry.register(ListFilesTool())
        registry.register(CalculatorTool())

        val systemPrompt = """You are a helpful assistant with access to tools.
When the user asks about files or directories, use the list_files tool to look up the actual contents.
When the user asks to calculate something, use the calculator tool.
Always use a tool when one is relevant — do not guess file listings."""

        val agentLoop = AgentLoop(
            runtime = runtime,
            template = template,
            toolRegistry = registry,
            eosTokenId = eosTokenId,
            config = AgentConfig(
                maxToolRounds = 5,
                maxTokensPerRound = maxTokens,
                temperature = temperature
            ),
            decode = { tokenId -> tokenizer.decode(tokenId) }
        )

        val messages = mutableListOf(
            ChatMessage(role = ChatRole.SYSTEM, content = systemPrompt)
        )

        println("KLlama Tool Calling Demo (type 'quit' to exit)")
        println("Available tools: ${registry.definitions().joinToString { it.name }}")
        println("---")
        println("Try: \"What files are in the current directory?\"")
        println("     \"List files in /tmp\"")
        println("     \"How many bytes is 1024 * 1024 * 512?\"")
        println("---")

        val listener = object : AgentListener {
            override fun onToken(token: String) {
                print(token)
                System.out.flush()
            }

            override fun onAssistantMessage(text: String) {
                println()
            }

            override fun onToolCalls(calls: List<ToolCall>) {
                for (call in calls) {
                    println("[Tool Call] ${call.name}(${call.arguments})")
                }
            }

            override fun onToolResult(call: ToolCall, result: String) {
                println("[Tool Result] ${call.name} -> $result")
                print("Assistant: ")
                System.out.flush()
            }

            override fun onComplete(finalResponse: String) {
                // Already printed via onToken
            }
        }

        while (true) {
            print("\nUser: ")
            System.out.flush()
            val input = readlnOrNull()?.trim() ?: break
            if (input.equals("quit", ignoreCase = true) || input.equals("exit", ignoreCase = true)) {
                break
            }
            if (input.isEmpty()) continue

            messages.add(ChatMessage(role = ChatRole.USER, content = input))

            print("Assistant: ")
            System.out.flush()

            agentLoop.runWithEncoder(
                messages = messages,
                encode = { text -> tokenizer.encode(text) },
                listener = listener
            )
        }

        println("Goodbye!")
    }
}

/**
 * Tool that lists files in a local directory.
 *
 * Returns file names, sizes, and whether each entry is a directory.
 * Restricted to readable directories and limits output to 50 entries.
 */
internal class ListFilesTool : Tool {

    override val definition: ToolDefinition = ToolDefinition(
        name = "list_files",
        description = "List files and directories in a local folder. Returns names, sizes, and types.",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("path") {
                    put("type", "string")
                    put("description", "Absolute or relative path to the directory to list, e.g. '/tmp' or '.'")
                }
            }
            putJsonArray("required") { add(JsonPrimitive("path")) }
        }
    )

    override fun execute(arguments: JsonObject): String {
        val path = arguments["path"]?.jsonPrimitive?.content
            ?: return "Error: missing 'path' argument"

        val dir = File(path)
        if (!dir.exists()) return "Error: '$path' does not exist"
        if (!dir.isDirectory) return "Error: '$path' is not a directory"
        if (!dir.canRead()) return "Error: '$path' is not readable"

        val entries = dir.listFiles() ?: return "Error: could not list '$path'"
        if (entries.isEmpty()) return "(empty directory)"

        val sorted = entries.sortedWith(compareBy({ !it.isDirectory }, { it.name }))
        val limited = sorted.take(50)
        val sb = StringBuilder()
        for (entry in limited) {
            val type = if (entry.isDirectory) "dir" else "file"
            val size = if (entry.isFile) " (${formatSize(entry.length())})" else ""
            sb.appendLine("[$type] ${entry.name}$size")
        }
        if (entries.size > 50) {
            sb.appendLine("... and ${entries.size - 50} more entries")
        }
        return sb.toString().trimEnd()
    }

    private fun formatSize(bytes: Long): String = when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        bytes < 1024 * 1024 * 1024 -> "${"%.1f".format(bytes / (1024.0 * 1024.0))} MB"
        else -> "${"%.1f".format(bytes / (1024.0 * 1024.0 * 1024.0))} GB"
    }
}
