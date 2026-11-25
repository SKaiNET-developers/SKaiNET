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
    fun testLoweringFlowsIntoJson() {
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
            annotation class Lowering(
                val composite: String = "",
                val decomposes: Boolean = true,
                val native: Boolean = false,
            )

            @InProgress("cpu", owner = "ops-team", issue = "GH-1")
            @Lowering(composite = "skainet.gelu")
            fun geluFn(): String = "ok"

            @InProgress("cpu", owner = "ops-team", issue = "GH-2")
            @Lowering(decomposes = false, native = true)
            fun tanhFn(): String = "ok"

            @InProgress("cpu", owner = "ops-team", issue = "GH-3")
            fun plainFn(): String = "ok"
        """.trimIndent()

        val source = SourceFile.kotlin("sk/ainet/lang/ops/TestLoweringOps.kt", sourceCode)
        val compilation = KotlinCompilation().apply {
            sources = listOf(source)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }
        val result = compilation.compile()
        assertTrue(result.messages.contains("Generated operators.json"), "Processor should generate operators.json")

        val operatorsJson = compilation.workingDir.walkTopDown().firstOrNull { it.isFile && it.name == "operators.json" }
        assertTrue(operatorsJson != null && operatorsJson.exists(), "operators.json should be present")
        val text = operatorsJson!!.readText()
        assertTrue(text.contains("\"lowering\": \"composite · decompose\""), "composite + default decompose expected: $text")
        assertTrue(text.contains("\"lowering\": \"native\""), "native-only expected: $text")
        assertTrue(Regex("\"lowering\"").findAll(text).count() == 2, "plainFn must not carry a lowering field: $text")
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

    @Test
    fun testNotImplementedMarkersFlowIntoStatusAndLayers() {
        val annotations = SourceFile.kotlin("sk/ainet/lang/ops/Annotations.kt", """
            package sk.ainet.lang.ops

            @Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class NotImplemented(vararg val backends: String, val reason: String = "", val issue: String = "")

            @Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class InProgress(vararg val backends: String, val owner: String = "", val issue: String = "")

            @Target(AnnotationTarget.CLASS)
            @Retention(AnnotationRetention.SOURCE)
            annotation class Backend(val id: String, val displayName: String = "", val internal: Boolean = false)
        """.trimIndent())
        val ops = SourceFile.kotlin("sk/ainet/lang/tensor/ops/TensorOps.kt", """
            package sk.ainet.lang.tensor.ops

            import sk.ainet.lang.ops.Backend
            import sk.ainet.lang.ops.InProgress
            import sk.ainet.lang.ops.NotImplemented

            interface TensorOps {
                fun add(): Int
                fun partial(): Int
                @NotImplemented(reason = "no kernel yet", issue = "#42")
                fun missing(): Int
            }

            @Backend(id = "cpu")
            class CpuOps : TensorOps {
                override fun add(): Int = 1
                @NotImplemented("cpu", reason = "FP32 only", issue = "#7")
                @InProgress("cpu")
                override fun partial(): Int = 2
                override fun missing(): Int = throw NotImplementedError()
            }
        """.trimIndent())
        val nn = SourceFile.kotlin("sk/ainet/lang/nn/Layers.kt", """
            package sk.ainet.lang.nn

            import sk.ainet.lang.ops.NotImplemented

            abstract class Module
            class Dense : Module()
            @NotImplemented(reason = "forward throws", issue = "#99")
            class Deconv : Module()
        """.trimIndent())

        val compilation = KotlinCompilation().apply {
            sources = listOf(annotations, ops, nn)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }
        compilation.compile()
        val text = compilation.workingDir.walkTopDown()
            .first { it.isFile && it.name == "operators.json" }.readText()

        fun statusOf(fn: String): String =
            Regex("\"name\": \"$fn\"[\\s\\S]*?\"statusByBackend\": \\{([^}]*)}").find(text)!!.groupValues[1]

        assertTrue(statusOf("add").contains("\"cpu\": \"implemented\""), text)
        // @NotImplemented on the override wins over @InProgress and over "overrides it"
        assertTrue(statusOf("partial").contains("\"cpu\": \"not_implemented\""), text)
        // @NotImplemented on the interface function, no backend list -> "*"
        assertTrue(statusOf("missing").contains("\"*\": \"not_implemented\""), text)
        // ...and it overrides a backend that overrides the function but still throws
        assertTrue(statusOf("missing").contains("\"cpu\": \"not_implemented\""), text)
        assertTrue(!text.contains("\"name\": \"Deconv\",\n"), "class-level layer marker must not become an operator: $text")
        assertTrue(text.contains("\"content\": \"#42\""), text)
        assertTrue(text.contains("\"content\": \"FP32 only\""), text)
        // Layers table
        assertTrue(text.contains("{\"name\": \"Dense\", \"packageName\": \"sk.ainet.lang.nn\", \"status\": \"implemented\"}"), text)
        assertTrue(text.contains("\"name\": \"Deconv\", \"packageName\": \"sk.ainet.lang.nn\", \"status\": \"not_implemented\", \"reason\": \"forward throws\", \"issue\": \"#99\""), text)
        assertTrue(!text.contains("\"name\": \"Module\""), "abstract Module must not be listed: $text")
    }

    @Test
    fun testDarcValidatedOnMetricComputeFlowsIntoMetrics() {
        // A metric is not a TensorOps function: its @DarcValidated sits on compute().
        val annotations = SourceFile.kotlin("sk/ainet/lang/ops/Annotations.kt", """
            package sk.ainet.lang.ops

            @Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class InProgress(vararg val backends: String, val owner: String = "", val issue: String = "")

            @Target(AnnotationTarget.FUNCTION)
            @Retention(AnnotationRetention.SOURCE)
            annotation class DarcValidated(
                val by: String,
                val on: String,
                val commit: String = "",
                val referencesChecked: Boolean = true,
            )

            // The processor writes operators.json only when it sees at least one operator.
            @InProgress("cpu")
            fun someOp(): String = "ok"
        """.trimIndent())
        val metrics = SourceFile.kotlin("sk/ainet/lang/nn/metrics/Metrics.kt", """
            package sk.ainet.lang.nn.metrics

            import sk.ainet.lang.ops.DarcValidated

            interface Metric {
                fun compute(): Double
            }

            abstract class CountingMetric : Metric

            class Reviewed : CountingMetric() {
                @DarcValidated(by = "Reviewer One", on = "2026-11-25", commit = "abc1234")
                override fun compute(): Double = 1.0
            }

            class Plain : Metric {
                override fun compute(): Double = 0.0
            }

            internal class Hidden : Metric {
                @DarcValidated(by = "Reviewer One", on = "2026-11-25")
                override fun compute(): Double = 0.0
            }
        """.trimIndent())

        val compilation = KotlinCompilation().apply {
            sources = listOf(annotations, metrics)
            configureKsp {}
            symbolProcessorProviders = mutableListOf(OperatorDocProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }
        compilation.compile()
        val text = compilation.workingDir.walkTopDown()
            .first { it.isFile && it.name == "operators.json" }.readText()

        assertTrue(
            text.contains(
                "{\"name\": \"Reviewed\", \"packageName\": \"sk.ainet.lang.nn.metrics\", \"validated\": true, " +
                    "\"validatedBy\": \"Reviewer One\", \"validatedOn\": \"2026-11-25\", " +
                    "\"validatedCommit\": \"abc1234\", \"referencesChecked\": true}"
            ),
            text,
        )
        assertTrue(
            text.contains("{\"name\": \"Plain\", \"packageName\": \"sk.ainet.lang.nn.metrics\", \"validated\": false}"),
            text,
        )
        assertTrue(!text.contains("\"name\": \"Hidden\""), "internal metric must not be listed: $text")
        assertTrue(!text.contains("\"name\": \"CountingMetric\""), "abstract metric must not be listed: $text")
        assertTrue(!text.contains("\"name\": \"Metric\""), "the interface must not be listed: $text")
    }
}
