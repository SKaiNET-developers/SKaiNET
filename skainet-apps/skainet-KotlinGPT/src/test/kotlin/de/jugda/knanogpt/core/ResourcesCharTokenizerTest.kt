package de.jugda.knanogpt.core

import knanogpt.core.CharTokenizer
import kotlin.test.Test
import kotlin.test.assertEquals
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.test.BeforeTest

class ResourcesCharTokenizerTest {

    private lateinit var textContent: String

    @BeforeTest
    fun setUp() {
        // Adjust the path according to your project structure and the location of the file
        val url = CharTokenizerTest::class.java.classLoader.getResource("input.txt")
        val uri = url?.toURI() ?: throw IllegalArgumentException("File not found in resources.")

        val path = Paths.get(uri)
        textContent = String(Files.readAllBytes(path))
    }

    @Test
    fun `encode and decode test using resource file`() {
        val tokenizer = CharTokenizer(textContent)
        val encoded = tokenizer.encode(textContent)
        val decoded = tokenizer.decode(encoded)
        assertEquals(textContent, decoded)
    }
}

