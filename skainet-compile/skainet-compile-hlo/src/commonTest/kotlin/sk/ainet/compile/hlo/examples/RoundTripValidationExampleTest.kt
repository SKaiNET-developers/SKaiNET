package sk.ainet.compile.hlo.examples

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import sk.ainet.compile.hlo.validation.RoundTripValidationResult

class RoundTripValidationExampleTest {
    
    @Test
    fun demonstrateValidation_completesSuccessfully() {
        val result = RoundTripValidationExample.demonstrateValidation()
        
        assertNotNull(result.generatedModule, "Should generate a module")
        assertNotNull(result.correctnessResult, "Should have correctness result")
        assertNotNull(result.roundTripResult, "Should have round-trip result")
        assertNotNull(result.performanceMetrics, "Should have performance metrics")
        
        // The generated module should have content
        assertTrue(result.generatedModule.content.isNotEmpty(), "Generated module should have content")
        
        // Performance metrics should be reasonable
        assertTrue(result.performanceMetrics.moduleSize > 0, "Module size should be positive")
        assertTrue(result.performanceMetrics.operationCount >= 0, "Operation count should be non-negative")
        
        // Print summary for manual inspection
        println(result.summary())
    }
    
    @Test
    fun demonstratePerformanceBenchmarking_completesSuccessfully() {
        val result = RoundTripValidationExample.demonstratePerformanceBenchmarking()
        
        assertNotNull(result, "Should have benchmark results")
        assertTrue(result.runs.isNotEmpty(), "Should have benchmark runs")
        
        // Print summary for manual inspection
        println(result.summary())
    }
    
    @Test
    fun demonstrateParsing_completesSuccessfully() {
        val result = RoundTripValidationExample.demonstrateParsing()
        
        assertNotNull(result, "Should have parsing result")
        assertTrue(result.originalContent.isNotEmpty(), "Should have original content")
        
        if (result.parseSuccess) {
            assertNotNull(result.parsedStructure, "Should have parsed structure on success")
            assertTrue(result.reconstructedContent.isNotEmpty(), "Should have reconstructed content")
        } else {
            assertTrue(result.errors.isNotEmpty(), "Should have errors on failure")
        }
        
        // Print summary for manual inspection
        println(result.summary())
    }
}