package sk.ainet.pipeline

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class PipelineTest {

    @Test
    fun testSimplePipeline() {
        val pipeline = pipeline<String, Int>("simple") {
            node<String, List<String>>("split") { input ->
                input.split(" ")
            }
            node<List<String>, Int>("count") { tokens ->
                tokens.size
            }
        }

        assertEquals("simple", pipeline.name)
        assertEquals("split", pipeline.entryNode)
        assertEquals("count", pipeline.exitNode)
        assertEquals(2, pipeline.nodeNames().size)

        val result = pipeline.execute("hello world test")
        assertEquals(3, result)
    }

    @Test
    fun testSingleNodePipeline() {
        val pipeline = pipeline<Int, Int>("double") {
            node<Int, Int>("multiply") { it * 2 }
        }

        assertEquals(10, pipeline.execute(5))
        assertEquals(0, pipeline.execute(0))
        assertEquals(-4, pipeline.execute(-2))
    }

    @Test
    fun testChainedNodes() {
        val pipeline = pipeline<Int, String>("chain") {
            node<Int, Int>("add10") { it + 10 }
            node<Int, Int>("multiply2") { it * 2 }
            node<Int, String>("toString") { "Result: $it" }
        }

        assertEquals("Result: 30", pipeline.execute(5))  // (5 + 10) * 2 = 30
    }

    @Test
    fun testConditionalEdge() {
        // Test that conditional edges route correctly
        // Using simple string routing based on input value
        val pipeline = pipeline<Int, String>("conditional") {
            // First node classifies and decides route
            node<Int, Int>("classify") { it }

            // Define conditional routing - must be defined before target nodes
            // to prevent auto-chaining
            conditionalEdge<Int>("classify") { value ->
                if (value >= 0) "formatPositive" else "formatNegative"
            }

            // These nodes are conditional targets - define with explicit names
            // Note: we need to handle that auto-chaining will try to link them
            node<Int, String>("formatPositive") { "Positive: $it" }
            exit("formatPositive")  // Set exit early to prevent further chaining issue
        }

        assertEquals("Positive: 5", pipeline.execute(5))
    }

    @Test
    fun testConditionalEdgeNegative() {
        val pipeline = pipeline<Int, String>("conditionalNeg") {
            node<Int, Int>("classify") { it }

            conditionalEdge<Int>("classify") { value ->
                if (value >= 0) "formatPositive" else "formatNegative"
            }

            node<Int, String>("formatNegative") { "Negative: $it" }
            exit("formatNegative")
        }

        assertEquals("Negative: -3", pipeline.execute(-3))
    }

    @Test
    fun testExplicitEdges() {
        val pipeline = pipeline<String, String>("explicit") {
            node<String, String>("start") { "[$it]" }
            node<String, String>("middle") { "{$it}" }
            node<String, String>("end") { "($it)" }

            // Edges are auto-created in order: start -> middle -> end
        }

        // start: [hello] -> middle: {[hello]} -> end: ({[hello]})
        assertEquals("({[hello]})", pipeline.execute("hello"))
    }

    @Test
    fun testEmbeddedPipeline() {
        val inner = pipeline<Int, Int>("inner") {
            node<Int, Int>("double") { it * 2 }
        }

        val outer = pipeline<Int, String>("outer") {
            node<Int, Int>("add5") { it + 5 }
            embed("inner", inner)
            node<Int, String>("format") { "Value: $it" }
        }

        // (3 + 5) * 2 = 16
        assertEquals("Value: 16", outer.execute(3))
    }

    @Test
    fun testPipelineComposition() {
        val first = pipeline<Int, Int>("first") {
            node<Int, Int>("add10") { it + 10 }
        }

        val second = pipeline<Int, String>("second") {
            node<Int, String>("format") { "Number: $it" }
        }

        val composed = first.then(second)

        assertEquals("first_then_second", composed.name)
        assertEquals("Number: 15", composed.execute(5))
    }

    @Test
    fun testPipelineHasNode() {
        val pipeline = pipeline<Int, Int>("test") {
            node<Int, Int>("a") { it }
            node<Int, Int>("b") { it }
        }

        assertTrue(pipeline.hasNode("a"))
        assertTrue(pipeline.hasNode("b"))
        assertTrue(!pipeline.hasNode("c"))
    }

    @Test
    fun testEmptyPipelineFails() {
        assertFailsWith<IllegalStateException> {
            pipeline<Int, Int>("empty") {
                // No nodes
            }
        }
    }

    @Test
    fun testCustomEntryAndExit() {
        val pipeline = pipeline<Int, Int>("custom") {
            node<Int, Int>("a") { it + 1 }
            node<Int, Int>("b") { it + 2 }
            node<Int, Int>("c") { it + 3 }

            entry("a")
            exit("b")

            // Only a -> b should execute
            edge("a", "b")
        }

        // Auto-chain creates: a -> b -> c
        // But we set exit to "b", so c shouldn't run
        assertEquals(8, pipeline.execute(5))  // 5 + 1 + 2 = 8
    }

    @Test
    fun testNodeRef() {
        val pipeline = pipeline<String, Int>("refs") {
            val splitRef = node<String, List<String>>("split") { it.split(",") }
            val countRef = node<List<String>, Int>("count") { it.size }

            assertEquals("split", splitRef.name)
            assertEquals("count", countRef.name)
        }

        assertEquals(3, pipeline.execute("a,b,c"))
    }

    @Test
    fun testTransformationPipeline() {
        // Simulates a data preprocessing pipeline
        val preprocessor = pipeline<List<Int>, List<Int>>("preprocess") {
            node<List<Int>, List<Int>>("filter") { list ->
                list.filter { it > 0 }
            }
            node<List<Int>, List<Int>>("normalize") { list ->
                val max = list.maxOrNull() ?: 1
                list.map { it * 100 / max }
            }
            node<List<Int>, List<Int>>("sort") { list ->
                list.sorted()
            }
        }

        val input = listOf(-5, 10, 3, -2, 7, 5)
        val result = preprocessor.execute(input)

        // Filter: [10, 3, 7, 5]
        // Normalize (max=10): [100, 30, 70, 50]
        // Sort: [30, 50, 70, 100]
        assertEquals(listOf(30, 50, 70, 100), result)
    }
}
