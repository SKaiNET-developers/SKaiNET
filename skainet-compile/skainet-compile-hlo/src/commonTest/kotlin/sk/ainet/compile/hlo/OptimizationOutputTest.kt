package sk.ainet.compile.hlo

import kotlin.test.*

/**
 * Test to see actual optimization output
 */
class OptimizationOutputTest {
    
    @Test
    fun debugConstantFolding() {
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
        
        println("=== CONSTANT FOLDING TEST ===")
        println("Original:")
        println(module.content)
        
        val optimizer = ConstantFoldingPass()
        val optimized = optimizer.apply(module)
        
        println("\nOptimized:")
        println(optimized.content)
        println("\nMetadata: ${optimized.metadata}")
        
        // Just check that optimization was applied
        assertTrue(optimized.metadata.containsKey("optimizations"))
    }
    
    @Test
    fun debugDeadCodeElimination() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<f32>) -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.0> : tensor<f32>
                    %2 = stablehlo.add %0, %1 : tensor<f32>
                    %3 = stablehlo.add %arg0, %0 : tensor<f32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        println("=== DEAD CODE ELIMINATION TEST ===")
        println("Original:")
        println(module.content)
        
        val optimizer = DeadCodeEliminationPass()
        val optimized = optimizer.apply(module)
        
        println("\nOptimized:")
        println(optimized.content)
        println("\nMetadata: ${optimized.metadata}")
        
        // Just check that optimization was applied
        assertTrue(optimized.metadata.containsKey("optimizations"))
    }
    
    @Test
    fun debugOperationFusion() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
                    %1 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
                    %2 = stablehlo.maximum %0, %1 : tensor<2x2xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        println("=== OPERATION FUSION TEST ===")
        println("Original:")
        println(module.content)
        
        val optimizer = OperationFusionPass()
        val optimized = optimizer.apply(module)
        
        println("\nOptimized:")
        println(optimized.content)
        println("\nMetadata: ${optimized.metadata}")
        
        // Just check that optimization was applied
        assertTrue(optimized.metadata.containsKey("optimizations"))
    }
}