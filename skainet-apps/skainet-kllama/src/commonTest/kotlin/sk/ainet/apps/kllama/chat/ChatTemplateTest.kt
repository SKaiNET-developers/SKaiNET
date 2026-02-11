package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertTrue
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

class ChatTemplateTest {

    private val systemMsg = ChatMessage(ChatRole.SYSTEM, "You are helpful.")
    private val userMsg = ChatMessage(ChatRole.USER, "Hello!")

    // --- Llama 3 Template Tests ---

    @Test
    fun llama3BasicFormat() {
        val template = Llama3ChatTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|begin_of_text|>")
        assertContains(result, "<|start_header_id|>system<|end_header_id|>")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|eot_id|>")
        assertContains(result, "<|start_header_id|>user<|end_header_id|>")
        assertContains(result, "Hello!")
        // Should end with assistant generation prompt
        assertTrue(result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
    }

    @Test
    fun llama3NoGenerationPrompt() {
        val template = Llama3ChatTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        assertTrue(result.endsWith("<|eot_id|>"))
    }

    @Test
    fun llama3WithTools() {
        val template = Llama3ChatTemplate()
        val tool = ToolDefinition(
            name = "calculator",
            description = "Evaluate math",
            parameters = buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    putJsonObject("expression") {
                        put("type", "string")
                    }
                }
            }
        )

        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        assertContains(result, "calculator")
        assertContains(result, "Evaluate math")
        assertContains(result, "\"name\"")
    }

    @Test
    fun llama3ToolRole() {
        val template = Llama3ChatTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "42", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|start_header_id|>tool<|end_header_id|>")
        assertContains(result, "42")
    }

    // --- ChatML Template Tests ---

    @Test
    fun chatMLBasicFormat() {
        val template = ChatMLTemplate()
        val result = template.apply(listOf(systemMsg, userMsg))

        assertContains(result, "<|im_start|>system")
        assertContains(result, "You are helpful.")
        assertContains(result, "<|im_end|>")
        assertContains(result, "<|im_start|>user")
        assertContains(result, "Hello!")
        // Should end with assistant generation prompt
        assertTrue(result.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun chatMLNoGenerationPrompt() {
        val template = ChatMLTemplate()
        val result = template.apply(listOf(userMsg), addGenerationPrompt = false)

        assertTrue(!result.endsWith("<|im_start|>assistant\n"))
        assertTrue(result.endsWith("<|im_end|>\n"))
    }

    @Test
    fun chatMLWithTools() {
        val template = ChatMLTemplate()
        val tool = ToolDefinition(
            name = "search",
            description = "Search the web",
            parameters = buildJsonObject { put("type", "object") }
        )

        val result = template.apply(listOf(userMsg), tools = listOf(tool))

        assertContains(result, "search")
        assertContains(result, "Search the web")
        assertContains(result, "<tool_call>")
    }

    @Test
    fun chatMLToolRole() {
        val template = ChatMLTemplate()
        val toolMsg = ChatMessage(ChatRole.TOOL, "result data", toolCallId = "call_0")
        val result = template.apply(listOf(toolMsg), addGenerationPrompt = false)

        assertContains(result, "<|im_start|>tool")
        assertContains(result, "result data")
    }
}
