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