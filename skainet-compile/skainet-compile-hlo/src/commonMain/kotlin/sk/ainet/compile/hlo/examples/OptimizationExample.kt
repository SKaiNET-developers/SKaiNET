package sk.ainet.compile.hlo.examples

import sk.ainet.compile.hlo.*

/**
 * Example demonstrating the optimization framework capabilities.
 * 
 * This example shows how to use the StableHLO optimization passes
 * to improve generated code through constant folding, dead code elimination,
 * and operation fusion.
 */
public object OptimizationExample {
    
    /**
     * Demonstrates basic optimization usage
     */
    public fun basicOptimizationExample(): String {
        // Create a sample MLIR module with optimization opportunities
        val unoptimizedModule = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<2x3xf32>
                    %1 = stablehlo.constant dense<2.0> : tensor<2x3xf32>
                    %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
                    %3 = stablehlo.add %arg0, %2 : tensor<2x3xf32>
                    %4 = stablehlo.constant dense<0.0> : tensor<2x3xf32>
                    %5 = stablehlo.maximum %3, %4 : tensor<2x3xf32>
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        // Apply optimizations
        val optimizer = StableHloOptimizer.createDefault()
        val optimizedModule = optimizer.optimize(unoptimizedModule)
        
        return """
            Original module:
            ${unoptimizedModule.content}
            
            Optimized module:
            ${optimizedModule.content}
            
            Applied optimizations: ${optimizedModule.metadata["optimizations"]}
        """.trimIndent()
    }
    
    /**
     * Demonstrates aggressive optimization with multiple passes
     */
    public fun aggressiveOptimizationExample(): String {
        val complexModule = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> () {
                    %0 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
                    %1 = stablehlo.constant dense<1.0> : tensor<4x4xf32>
                    %2 = stablehlo.multiply %0, %1 : tensor<4x4xf32>
                    %3 = stablehlo.add %arg0, %arg1 : tensor<4x4xf32>
                    %4 = stablehlo.add %3, %2 : tensor<4x4xf32>
                    %5 = stablehlo.constant dense<2.0> : tensor<4x4xf32>
                    %6 = stablehlo.multiply %4, %5 : tensor<4x4xf32>
                    %7 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
                    %8 = stablehlo.maximum %6, %7 : tensor<4x4xf32>
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        // Apply aggressive optimizations
        val optimizer = StableHloOptimizer.createAggressive()
        val optimizedModule = optimizer.optimize(complexModule)
        
        return """
            Complex module with multiple optimization opportunities:
            ${complexModule.content}
            
            After aggressive optimization:
            ${optimizedModule.content}
            
            Applied optimizations: ${optimizedModule.metadata["optimizations"]}
        """.trimIndent()
    }
    
    /**
     * Demonstrates custom optimization pipeline
     */
    public fun customOptimizationExample(): String {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<3x3xf32>) -> () {
                    %0 = stablehlo.constant dense<5.0> : tensor<3x3xf32>
                    %1 = stablehlo.constant dense<3.0> : tensor<3x3xf32>
                    %2 = stablehlo.add %0, %1 : tensor<3x3xf32>
                    %3 = stablehlo.add %arg0, %2 : tensor<3x3xf32>
                    %4 = stablehlo.multiply %3, %1 : tensor<3x3xf32>
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        // Create custom optimization pipeline
        val customOptimizer = StableHloOptimizer().apply {
            addPass(ConstantFoldingPass())
            addPass(OperationFusionPass())
            addPass(DeadCodeEliminationPass())
        }
        
        val optimizedModule = customOptimizer.optimize(module)
        
        return """
            Custom optimization pipeline example:
            
            Original:
            ${module.content}
            
            After custom optimization:
            ${optimizedModule.content}
            
            Applied optimizations: ${optimizedModule.metadata["optimizations"]}
        """.trimIndent()
    }
    
    /**
     * Demonstrates optimization benefits analysis
     */
    public fun optimizationBenefitsAnalysis(): String {
        val testModule = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x2xf32>) -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
                    %1 = stablehlo.constant dense<1.0> : tensor<2x2xf32>
                    %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
                    %3 = stablehlo.add %arg0, %2 : tensor<2x2xf32>
                    %4 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
                    %5 = stablehlo.maximum %3, %4 : tensor<2x2xf32>
                    %6 = stablehlo.constant dense<10.0> : tensor<2x2xf32>
                    %7 = stablehlo.multiply %6, %6 : tensor<2x2xf32>
                    return
                  }
                }
            """.trimIndent(),
            functionName = "main"
        )
        
        val originalOpCount = countOperations(testModule.content)
        
        val optimizer = StableHloOptimizer.createDefault()
        val optimizedModule = optimizer.optimize(testModule)
        val optimizedOpCount = countOperations(optimizedModule.content)
        
        val reduction = ((originalOpCount - optimizedOpCount).toFloat() / originalOpCount * 100).toInt()
        
        return """
            Optimization Benefits Analysis:
            
            Original operation count: $originalOpCount
            Optimized operation count: $optimizedOpCount
            Reduction: $reduction%
            
            Optimizations applied: ${optimizedModule.metadata["optimizations"]}
            
            Benefits:
            - Reduced memory traffic through constant folding
            - Eliminated unused computations
            - Fused operations for better cache locality
            - Simplified control flow
        """.trimIndent()
    }
    
    private fun countOperations(mlirContent: String): Int {
        return mlirContent.lines()
            .count { line -> 
                line.trim().contains("stablehlo.") && 
                !line.trim().startsWith("//") 
            }
    }
}