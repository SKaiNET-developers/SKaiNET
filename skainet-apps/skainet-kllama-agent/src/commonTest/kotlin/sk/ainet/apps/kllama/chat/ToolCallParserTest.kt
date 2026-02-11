package sk.ainet.apps.kllama.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ToolCallParserTest {

    // --- Hermes-style parsing ---

    @Test
    fun parseHermesStyleToolCall() {
        val text = """
            Some thinking text...
            <tool_call>
            {"name": "calculator", "arguments": {"expression": "2 + 3"}}
            </tool_call>
        """.trimIndent()

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("2 + 3", calls[0].arguments["expression"]?.toString()?.trim('"'))
    }

    @Test
    fun parseMultipleHermesToolCalls() {
        val text = """
            <tool_call>{"name": "search", "arguments": {"query": "weather"}}</tool_call>
            <tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>
        """.trimIndent()

        val calls = ToolCallParser.parse(text)
        assertEquals(2, calls.size)
        assertEquals("search", calls[0].name)
        assertEquals("calculator", calls[1].name)
    }

    // --- Llama 3.1-style parsing ---

    @Test
    fun parseLlama31StyleToolCall() {
        val text = """{"name": "calculator", "arguments": {"expression": "5 * 10"}}"""

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("calculator", calls[0].name)
        assertEquals("5 * 10", calls[0].arguments["expression"]?.toString()?.trim('"'))
    }

    @Test
    fun parseLlama31StyleWithId() {
        val text = """{"id": "call_42", "name": "search", "arguments": {"query": "Kotlin"}}"""

        val calls = ToolCallParser.parse(text)
        assertEquals(1, calls.size)
        assertEquals("call_42", calls[0].id)
        assertEquals("search", calls[0].name)
    }

    // --- No tool call ---

    @Test
    fun parsePlainTextReturnsEmpty() {
        val text = "This is just a normal response with no tool calls."
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseEmptyStringReturnsEmpty() {
        val calls = ToolCallParser.parse("")
        assertTrue(calls.isEmpty())
    }

    @Test
    fun parseMalformedJsonReturnsEmpty() {
        val text = """{"name": "calculator", "arguments": {invalid json}}"""
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }

    // --- containsToolCall ---

    @Test
    fun containsToolCallDetectsHermesFormat() {
        assertTrue(ToolCallParser.containsToolCall("<tool_call>{}</tool_call>"))
    }

    @Test
    fun containsToolCallDetectsJsonFormat() {
        assertTrue(ToolCallParser.containsToolCall("""{"name": "calc", "arguments": {}}"""))
    }

    @Test
    fun containsToolCallReturnsFalseForPlainText() {
        assertFalse(ToolCallParser.containsToolCall("Hello, world!"))
    }

    // --- Edge cases ---

    @Test
    fun parseMissingArgumentsField() {
        val text = """{"name": "no_args"}"""
        val calls = ToolCallParser.parse(text)
        // Should still parse — missing arguments defaults to empty
        assertEquals(1, calls.size)
        assertEquals("no_args", calls[0].name)
        assertTrue(calls[0].arguments.isEmpty())
    }

    @Test
    fun parseMissingNameReturnsEmpty() {
        val text = """{"arguments": {"expression": "1+1"}}"""
        val calls = ToolCallParser.parse(text)
        assertTrue(calls.isEmpty())
    }
}
