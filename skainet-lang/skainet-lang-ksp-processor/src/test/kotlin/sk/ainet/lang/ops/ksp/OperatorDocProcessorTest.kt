@file:OptIn(ExperimentalCompilerApi::class)

package sk.ainet.lang.ops.ksp

import com.tschuchort.compiletesting.KotlinCompilation
import com.tschuchort.compiletesting.SourceFile
import com.tschuchort.compiletesting.configureKsp
import com.tschuchort.compiletesting.symbolProcessorProviders
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.junit.Test
import kotlin.test.assertTrue

class OperatorDocProcessorTest {

    @Test
    fun testInProgressAnnotationProcessing() {
        val sourceCode = """
            package test
            
            // Define the annotation inline to ensure it's available
            @Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class InProgress(
                vararg val backends: String,
                val owner: String = "",
                val issue: String = ""
            )
            
            // Simple test function instead of complex class hierarchy
            @InProgress("Metal", owner="ops-team", issue="GH-1234")
            fun testFunction(): String {
                return "test"
            }
            
            @InProgress("CPU", owner="cpu-team", issue="GH-5678")  
            fun anotherTestFunction(): Int {
                return 42
            }
        """.trimIndent()

        val source = SourceFile.kotlin("test/TestTensorOps.kt", sourceCode)
        
        val compilation = KotlinCompilation().apply {
            sources = listOf(source)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }

        val result = compilation.compile()
        val output = result.messages

        println("[DEBUG_LOG] Compilation result: ${result.exitCode}")
        println("[DEBUG_LOG] Output messages: $output")

        // Check if the processor found the InProgress annotation
        assertTrue(output.contains("Found 2 annotated symbols"),
            "Processor should find the @InProgress annotated functions")
        
        // Check if JSON output was generated
        assertTrue(output.contains("Generated operators.json"), 
            "Processor should generate operators.json file")
        
        // Since the processor found the annotations, the test passes
        // The detailed annotation processing would require more complex test setup
        // but the key requirement is that the test compiles and runs
    }

    @Test
    fun testDarcValidatedFlowsIntoJson() {
        // Two functions annotated with @InProgress so the processor picks
        // them up via the annotation-discovery path. Only one of them
        // carries @DarcValidated — we expect that one (and only that one)
        // to land in operators.json with the validation block.
        val sourceCode = """
            package sk.ainet.lang.ops

            @Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class InProgress(
                vararg val backends: String,
                val owner: String = "",
                val issue: String = ""
            )

            @Target(AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class DarcValidated(
                val by: String,
                val on: String,
                val commit: String = "",
                val referencesChecked: Boolean = true,
            )

            @InProgress("cpu", owner = "ops-team", issue = "GH-1")
            @DarcValidated(by = "Reviewer One", on = "2026-05-24")
            fun validatedFn(): String = "ok"

            @InProgress("cpu", owner = "ops-team", issue = "GH-2")
            fun plainFn(): String = "ok"
        """.trimIndent()

        val source = SourceFile.kotlin("sk/ainet/lang/ops/TestDarcOps.kt", sourceCode)

        val compilation = KotlinCompilation().apply {
            sources = listOf(source)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }

        val result = compilation.compile()
        val output = result.messages
        println("[DEBUG_LOG] Compilation result: ${result.exitCode}")
        println("[DEBUG_LOG] Output messages: $output")

        assertTrue(output.contains("Generated operators.json"),
            "Processor should generate operators.json")

        // Locate the JSON the processor just wrote. KSP code generator
        // outputs resources somewhere under the compilation's working
        // dir; the layout is version-dependent, so search rather than
        // hard-code a path.
        val operatorsJson = compilation.workingDir.walkTopDown()
            .firstOrNull { it.isFile && it.name == "operators.json" }
        assertTrue(operatorsJson != null && operatorsJson.exists(),
            "operators.json should be present in compilation output")
        val text = operatorsJson.readText()
        println("[DEBUG_LOG] operators.json contents: $text")

        assertTrue(text.contains("\"validatedFn\""),
            "validatedFn should be present in JSON")
        assertTrue(text.contains("\"validated\": true"),
            "validated=true should be emitted for the annotated function")
        assertTrue(text.contains("\"validatedBy\": \"Reviewer One\""),
            "validatedBy should carry the reviewer identity")
        assertTrue(text.contains("\"validatedOn\": \"2026-05-24\""),
            "validatedOn should carry the ISO date")

        // The unannotated function must NOT carry a validation block.
        // The processor only emits the keys when validated=true, so a
        // single occurrence in the file is the expected count.
        val validatedKeyCount = Regex("\"validated\":\\s*true").findAll(text).count()
        assertTrue(validatedKeyCount == 1,
            "Exactly one function should emit validated=true (got $validatedKeyCount)")
    }

    @Test
    fun testDslOpAnnotationProcessing() {
        val sourceCode = """
            package sk.ainet.lang.ops
            
            @Target(AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class DslOp(
                val category: String = "",
                val description: String = ""
            )
            
            @DslOp(category = "Similarity", description = "Calculates cosine distance")
            fun cosineDistance(): Float {
                return 0.5f
            }
        """.trimIndent()

        val source = SourceFile.kotlin("sk/ainet/lang/ops/TestDslOps.kt", sourceCode)
        
        val compilation = KotlinCompilation().apply {
            sources = listOf(source)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }

        val result = compilation.compile()
        val output = result.messages

        println("[DEBUG_LOG] Compilation result: ${result.exitCode}")
        println("[DEBUG_LOG] Output messages: $output")

        // Check if the processor found the DslOp annotation
        assertTrue(output.contains("Found 1 annotated symbols"),
            "Processor should find the @DslOp annotated function")
        
        // Check if JSON output was generated
        assertTrue(output.contains("Generated operators.json"), 
            "Processor should generate operators.json file")
    }
}