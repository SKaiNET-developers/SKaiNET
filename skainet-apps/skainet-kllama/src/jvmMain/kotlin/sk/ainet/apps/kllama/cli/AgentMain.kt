package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.models.llama.LlamaRuntimeInterface
import sk.ainet.apps.kllama.chat.*
import sk.ainet.apps.kllama.agent.generateUntilStop
import sk.ainet.lang.types.DType
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.jsonPrimitive

/**
 * Interactive chat and agent CLI for KLlama.
 *
 * Provides two modes:
 * - **Chat mode**: Interactive multi-turn conversation with the model.
 * - **Agent mode**: Interactive conversation with tool calling support.
 */
public class AgentCli<T : DType>(
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
     * Run interactive chat mode (no tool calling).
     */
    public fun runChat(
        systemPrompt: String = "You are a helpful assistant.",
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ) {
        val messages = mutableListOf(
            ChatMessage(role = ChatRole.SYSTEM, content = systemPrompt)
        )

        println("KLlama Chat Mode (type 'quit' to exit)")
        println("System: $systemPrompt")
        println("---")

        while (true) {
            print("\nUser: ")
            System.out.flush()
            val input = readlnOrNull()?.trim() ?: break
            if (input.equals("quit", ignoreCase = true) || input.equals("exit", ignoreCase = true)) {
                break
            }
            if (input.isEmpty()) continue

            messages.add(ChatMessage(role = ChatRole.USER, content = input))

            runtime.reset()
            val prompt = template.apply(messages, emptyList(), addGenerationPrompt = true)
            val promptTokens = tokenizer.encode(prompt)

            print("Assistant: ")
            System.out.flush()

            val result = runtime.generateUntilStop(
                prompt = promptTokens,
                maxTokens = maxTokens,
                eosTokenId = eosTokenId,
                temperature = temperature,
                onToken = { tokenId ->
                    print(tokenizer.decode(tokenId))
                    System.out.flush()
                },
                decode = { tokenId -> tokenizer.decode(tokenId) }
            )

            println()
            messages.add(ChatMessage(role = ChatRole.ASSISTANT, content = result.text))
        }

        println("Goodbye!")
    }

    /**
     * Run interactive agent mode with tool calling.
     */
    public fun runAgent(
        systemPrompt: String = "You are a helpful assistant with access to tools.",
        maxTokens: Int = 512,
        temperature: Float = 0.7f
    ) {
        val registry = ToolRegistry()
        registry.register(CalculatorTool())

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

        println("KLlama Agent Mode (type 'quit' to exit)")
        println("Available tools: ${registry.definitions().joinToString { it.name }}")
        println("System: $systemPrompt")
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
 * Demo calculator tool for testing the agent loop.
 */
internal class CalculatorTool : Tool {

    override val definition: ToolDefinition = ToolDefinition(
        name = "calculator",
        description = "Evaluate a mathematical expression. Supports +, -, *, / and parentheses.",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("expression") {
                    put("type", "string")
                    put("description", "The mathematical expression to evaluate, e.g. '2 + 3 * 4'")
                }
            }
            putJsonArray("required") { add(JsonPrimitive("expression")) }
        }
    )

    override fun execute(arguments: JsonObject): String {
        val expression = arguments["expression"]?.jsonPrimitive?.content
            ?: return "Error: missing 'expression' argument"
        return try {
            val result = evaluateExpression(expression)
            result.toString()
        } catch (e: Exception) {
            "Error: ${e.message}"
        }
    }

    /**
     * Simple expression evaluator supporting +, -, *, / and parentheses.
     */
    private fun evaluateExpression(expr: String): Double {
        val tokens = tokenize(expr)
        val parser = ExprParser(tokens)
        return parser.parseExpression()
    }

    private fun tokenize(expr: String): List<String> {
        val tokens = mutableListOf<String>()
        var i = 0
        while (i < expr.length) {
            when {
                expr[i].isWhitespace() -> i++
                expr[i].isDigit() || expr[i] == '.' -> {
                    val start = i
                    while (i < expr.length && (expr[i].isDigit() || expr[i] == '.')) i++
                    tokens.add(expr.substring(start, i))
                }
                expr[i] in "+-*/()" -> {
                    tokens.add(expr[i].toString())
                    i++
                }
                else -> error("Unexpected character: ${expr[i]}")
            }
        }
        return tokens
    }

    private class ExprParser(private val tokens: List<String>) {
        private var pos = 0

        fun parseExpression(): Double {
            var result = parseTerm()
            while (pos < tokens.size && tokens[pos] in listOf("+", "-")) {
                val op = tokens[pos++]
                val right = parseTerm()
                result = if (op == "+") result + right else result - right
            }
            return result
        }

        private fun parseTerm(): Double {
            var result = parseFactor()
            while (pos < tokens.size && tokens[pos] in listOf("*", "/")) {
                val op = tokens[pos++]
                val right = parseFactor()
                result = if (op == "*") result * right else result / right
            }
            return result
        }

        private fun parseFactor(): Double {
            if (pos < tokens.size && tokens[pos] == "(") {
                pos++ // skip (
                val result = parseExpression()
                if (pos < tokens.size && tokens[pos] == ")") pos++ // skip )
                return result
            }
            if (pos < tokens.size && tokens[pos] == "-") {
                pos++
                return -parseFactor()
            }
            return tokens[pos++].toDouble()
        }
    }
}
