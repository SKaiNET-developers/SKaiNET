package sk.ainet.compile.c

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Test for CCodegenFacade functionality.
 * 
 * This test verifies that the facade correctly integrates CCodeGenerator
 * and ArduinoLibraryPackager following the ModelExportFacade pattern.
 */
class CCodegenFacadeTest {
    
    @Test
    fun testFacadeCreation() {
        val facade = CCodegenFacade()
        assertNotNull(facade)
    }
    
    @Test
    fun testExportGraphToArduinoLibraryWithEmptyGraph() {
        val facade = CCodegenFacade()
        val graph = DefaultComputeGraph()
        
        try {
            facade.exportGraphToArduinoLibrary(
                graph = graph,
                outputPath = "/tmp/test",
                libraryName = "TestLibrary"
            )
            // Should fail with empty graph
            assertTrue(false, "Expected exception for empty graph")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message?.contains("no input nodes") == true || 
                      e.message?.contains("validation failed") == true)
        }
    }
    
    @Test
    fun testExportToArduinoLibraryWithUnsupportedModel() {
        val facade = CCodegenFacade()
        val unsupportedModel = "This is not a ComputeGraph"
        
        try {
            facade.exportToArduinoLibrary(
                model = unsupportedModel,
                outputPath = "/tmp/test",
                libraryName = "TestLibrary"
            )
            // Should fail with unsupported model type
            assertTrue(false, "Expected exception for unsupported model")
        } catch (e: IllegalStateException) {
            assertTrue(e.message?.contains("does not have a direct adapter") == true)
        }
    }
    
}