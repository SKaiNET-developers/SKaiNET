package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import sk.ainet.compile.hlo.validation.*

class MlirTestUtilitiesTest {
    
    private val testUtilities = MlirTestUtilities()
    
    @Test
    fun verifyCorrectness_validModule_returnsCorrect() {
        val validModule = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        val result = testUtilities.verifyCorrectness(validModule)
        
        assertTrue(result.syntaxValid, "Valid module should have correct syntax")
        assertTrue(result.semanticValid, "Valid module should be semantically correct")
        // Note: roundTripValid might be false depending on parser implementation
        assertTrue(result.errors.isEmpty() || result.errors.all { it.startsWith("RoundTrip:") }, 
                  "Should have no syntax/semantic errors")
    }
    
    @Test
    fun verifyCorrectness_invalidModule_returnsIncorrect() {
        val invalidModule = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0 // Missing operand and type
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        val result = testUtilities.verifyCorrectness(invalidModule)
        
        assertFalse(result.isCorrect, "Invalid module should not be correct")
        assertTrue(result.errors.isNotEmpty(), "Should have errors")
    }
    
    @Test
    fun verifyOperations_expectedOperationsPresent_returnsNoErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    %1 = stablehlo.relu %0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifyOperations(module, listOf("add", "relu"))
        
        assertTrue(errors.isEmpty(), "Should find all expected operations")
    }
    
    @Test
    fun verifyOperations_missingOperations_returnsErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifyOperations(module, listOf("add", "multiply", "relu"))
        
        assertEquals(2, errors.size, "Should report 2 missing operations")
        assertTrue(errors.any { it.contains("multiply") }, "Should report missing multiply")
        assertTrue(errors.any { it.contains("relu") }, "Should report missing relu")
    }
    
    @Test
    fun verifySSAForm_validSSA_returnsNoErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    %1 = stablehlo.relu %0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifySSAForm(module)
        
        assertTrue(errors.isEmpty(), "Valid SSA form should have no errors")
    }
    
    @Test
    fun verifySSAForm_redefinedValue_returnsError() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    %0 = stablehlo.relu %arg0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifySSAForm(module)
        
        assertTrue(errors.isNotEmpty(), "Redefined SSA value should cause error")
        assertTrue(errors.any { it.contains("redefined") }, "Should report redefinition")
    }
    
    @Test
    fun verifySSAForm_undefinedValue_returnsError() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %undefined : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifySSAForm(module)
        
        assertTrue(errors.isNotEmpty(), "Undefined SSA value should cause error")
        assertTrue(errors.any { it.contains("Undefined") }, "Should report undefined value")
    }
    
    @Test
    fun verifyTypeAnnotations_validTypes_returnsNoErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifyTypeAnnotations(module)
        
        assertTrue(errors.isEmpty(), "Valid type annotations should have no errors")
    }
    
    @Test
    fun verifyTypeAnnotations_missingTypeAnnotation_returnsError() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0
                    return
                  }
                }
            """.trimIndent()
        )
        
        val errors = testUtilities.verifyTypeAnnotations(module)
        
        assertTrue(errors.isNotEmpty(), "Missing type annotation should cause error")
        assertTrue(errors.any { it.contains("Missing type annotation") }, "Should report missing annotation")
    }
    
    @Test
    fun createSimpleTestGraph_returnsValidGraph() {
        val graph = testUtilities.createSimpleTestGraph()
        
        assertNotNull(graph, "Should create a valid graph")
        
        val nodes = graph.getTopologicalOrder()
        assertTrue(nodes.isNotEmpty(), "Graph should have nodes")
        
        // Should have input nodes
        val inputNodes = nodes.filter { it.operation.type == "input" || it.operation.name == "input" }
        assertTrue(inputNodes.isNotEmpty(), "Should have input nodes")
        
        // Should have operation nodes
        val opNodes = nodes.filter { it.operation.type != "input" && it.operation.name != "input" }
        assertTrue(opNodes.isNotEmpty(), "Should have operation nodes")
    }
    
    @Test
    fun validateMlirPatterns_expectedPatternsPresent_returnsNoErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val patterns = listOf("module {", "func.func @main", "stablehlo.add", "return")
        val errors = testUtilities.validateMlirPatterns(module, patterns)
        
        assertTrue(errors.isEmpty(), "All expected patterns should be found")
    }
    
    @Test
    fun validateMlirPatterns_missingPatterns_returnsErrors() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val patterns = listOf("stablehlo.multiply", "stablehlo.relu")
        val errors = testUtilities.validateMlirPatterns(module, patterns)
        
        assertEquals(2, errors.size, "Should report 2 missing patterns")
    }
    
    @Test
    fun extractPerformanceMetrics_returnsValidMetrics() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                    %1 = stablehlo.relu %0 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val metrics = testUtilities.extractPerformanceMetrics(module)
        
        assertTrue(metrics.containsKey("content_length"), "Should have content length")
        assertTrue(metrics.containsKey("line_count"), "Should have line count")
        assertTrue(metrics.containsKey("operation_count"), "Should have operation count")
        assertTrue(metrics.containsKey("ssa_value_count"), "Should have SSA value count")
        
        val operationCount = metrics["operation_count"] as Int
        assertEquals(2, operationCount, "Should count 2 operations")
        
        val ssaValueCount = metrics["ssa_value_count"] as Int
        assertTrue(ssaValueCount >= 2, "Should have at least 2 SSA values")
    }
}