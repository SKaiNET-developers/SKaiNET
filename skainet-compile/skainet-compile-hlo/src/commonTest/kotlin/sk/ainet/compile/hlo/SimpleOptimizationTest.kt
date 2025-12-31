package sk.ainet.compile.hlo

import kotlin.test.*

/**
 * Simple test to verify optimization functionality
 */
class SimpleOptimizationTest {
    
    @Test
    fun testBasicConstantFolding() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main() -> () {
                    %0 = stablehlo.constant dense<2.0> : tensor<f32>
                    %1 = stablehlo.constant dense<3.0> : tensor<f32>
                    %2 = stablehlo.add %0, %1 : tensor<f32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = ConstantFoldingPass()
        val optimized = optimizer.apply(module)
        
        // Should fold 2.0 + 3.0 = 5.0
        assertTrue(optimized.content.contains("dense<5.0>"), "Should contain folded constant 5.0")
        assertTrue(optimized.metadata.containsKey("optimizations"), "Should track optimizations")
    }
    
    @Test
    fun testBasicDeadCodeElimination() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<f32>) -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.0> : tensor<f32>
                    %2 = stablehlo.add %arg0, %0 : tensor<f32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = DeadCodeEliminationPass()
        val optimized = optimizer.apply(module)
        
        // %1 should be eliminated as it's unused
        assertFalse(optimized.content.contains("dense<2.0>"), "Should eliminate unused constant")
        assertTrue(optimized.content.contains("dense<1.0>"), "Should keep used constant")
    }
    
    @Test
    fun testFullOptimizationPipeline() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x2xf32>) -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
                    %1 = stablehlo.constant dense<2.0> : tensor<2x2xf32>
                    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
                    %3 = stablehlo.add %arg0, %2 : tensor<2x2xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = StableHloOptimizer.createDefault()
        val optimized = optimizer.optimize(module)
        
        // Should fold constants: 1.0 + 2.0 = 3.0
        assertTrue(optimized.content.contains("dense<3.0>"), "Should fold constants")
        
        // Should track all applied optimizations
        val appliedOpts = optimized.metadata["optimizations"] as? List<String> ?: emptyList()
        assertTrue(appliedOpts.isNotEmpty(), "Should have applied optimizations")
    }
}