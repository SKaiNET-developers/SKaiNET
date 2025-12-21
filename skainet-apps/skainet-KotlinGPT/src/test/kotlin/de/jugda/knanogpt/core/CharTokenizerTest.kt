package de.jugda.knanogpt.core

import knanogpt.core.CharTokenizer
import kotlin.test.Test
import kotlin.test.assertEquals


class CharTokenizerTest {

    @Test
    fun `encode test`() {
        val tokenizer = CharTokenizer("hello world")
        val encoded = tokenizer.encode("hello")
        assertEquals(listOf(3, 2, 4, 4, 5), encoded)
    }

    @Test
    fun `decode test`() {
        val tokenizer = CharTokenizer("hello world")
        val decoded = tokenizer.decode(listOf(3, 2, 4, 4, 5))
        assertEquals("hello", decoded)
    }

    @Test
    fun `encode and decode test`() {
        val text = "hello world"
        val tokenizer = CharTokenizer(text)
        val encoded = tokenizer.encode(text)
        val decoded = tokenizer.decode(encoded)
        assertEquals(text, decoded)
    }

    @Test
    fun `encode with characters not in tokenizer should fail`() {
        val tokenizer = CharTokenizer("hello")
        try {
            tokenizer.encode("world")
        } catch (e: IllegalStateException) {
            assertEquals("Character not found", e.message)
        }
    }

    @Test
    fun `decode with invalid tokens should fail`() {
        val tokenizer = CharTokenizer("hello")
        try {
            tokenizer.decode(listOf(10, 11))
        } catch (e: IllegalStateException) {
            assertEquals("Token not found", e.message)
        }
    }
}