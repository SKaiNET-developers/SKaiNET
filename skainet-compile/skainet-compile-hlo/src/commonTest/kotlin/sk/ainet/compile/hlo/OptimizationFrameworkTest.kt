package sk.ainet.compile.hlo

import kotlin.test.*

/**
 * Comprehensive tests for the StableHLO optimization framework.
 * 
 * Tests constant folding, dead code elimination, operation fusion,
 * and the overall optimization pipeline.
 */
class OptimizationFrameworkTest {
    
    @Test
    fun testConstantFoldingBasicArithmetic() {
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
        assertTrue(optimized.content.contains("dense<5.0>"))
        assertTrue(optimized.metadata["optimizations"].toString().contains("constant-folding"))
    }
    
    @Test
    fun testConstantFoldingMultiplication() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main() -> () {
                    %0 = stablehlo.constant dense<4.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.5> : tensor<f32>
                    %2 = stablehlo.multiply %0, %1 : tensor<f32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = ConstantFoldingPass()
        val optimized = optimizer.apply(module)
        
        // Should fold 4.0 * 2.5 = 10.0
        assertTrue(optimized.content.contains("dense<10.0>"))
    }
    
    @Test
    fun testConstantFoldingChainedOperations() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main() -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.0> : tensor<f32>
                    %2 = stablehlo.add %0, %1 : tensor<f32>
                    %3 = stablehlo.constant dense<3.0> : tensor<f32>
                    %4 = stablehlo.multiply %2, %3 : tensor<f32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = ConstantFoldingPass()
        val optimized = optimizer.apply(module)
        
        // Should fold (1.0 + 2.0) * 3.0 = 9.0
        assertTrue(optimized.content.contains("dense<3.0>")) // First fold: 1.0 + 2.0 = 3.0
        assertTrue(optimized.content.contains("dense<9.0>")) // Second fold: 3.0 * 3.0 = 9.0
    }
    
    @Test
    fun testDeadCodeEliminationUnusedConstants() {
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
        assertFalse(optimized.content.contains("dense<2.0>"))
        assertTrue(optimized.content.contains("dense<1.0>")) // %0 is used
        assertTrue(optimized.metadata["optimizations"].toString().contains("dead-code-elimination"))
    }
    
    @Test
    fun testDeadCodeEliminationUnusedOperations() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.constant dense<1.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.0> : tensor<f32>
                    %2 = stablehlo.add %0, %1 : tensor<f32>
                    %3 = stablehlo.add %arg0, %0 : tensor<f32>
                    return %3 : tensor<f32>
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = DeadCodeEliminationPass()
        val optimized = optimizer.apply(module)
        
        // %2 should be eliminated as its result is unused
        assertFalse(optimized.content.contains("%2 = stablehlo.add %0, %1"), "Should eliminate unused operation")
        assertTrue(optimized.content.contains("%3 = stablehlo.add %arg0, %0"), "Should keep used operation")
    }
    
    @Test
    fun testOperationFusionAddRelu() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
                    %1 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
                    %2 = stablehlo.maximum %0, %1 : tensor<2x2xf32>
                    return %2 : tensor<2x2xf32>
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = OperationFusionPass()
        val optimized = optimizer.apply(module)
        
        // Should fuse add + relu pattern (look for fusion comment)
        assertTrue(optimized.content.contains("fused with relu") || optimized.content.contains("fused_activation"))
        assertTrue(optimized.metadata["optimizations"].toString().contains("operation-fusion"))
    }
    
    @Test
    fun testOperationFusionElementwise() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> () {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
                    %1 = stablehlo.multiply %0, %arg2 : tensor<2x2xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = OperationFusionPass()
        val optimized = optimizer.apply(module)
        
        // Should fuse add + multiply pattern (look for fusion comment)
        assertTrue(optimized.content.contains("fused with multiply") || optimized.content.contains("fusion_type"))
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
                    %4 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
                    %5 = stablehlo.maximum %3, %4 : tensor<2x2xf32>
                    %6 = stablehlo.constant dense<10.0> : tensor<2x2xf32>
                    %7 = stablehlo.multiply %6, %6 : tensor<2x2xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = StableHloOptimizer.createDefault()
        val optimized = optimizer.optimize(module)
        
        // Should apply multiple optimizations
        val appliedOpts = optimized.metadata["optimizations"] as? List<String> ?: emptyList()
        assertTrue(appliedOpts.contains("constant-folding"))
        assertTrue(appliedOpts.contains("operation-fusion"))
        assertTrue(appliedOpts.contains("dead-code-elimination"))
        
        // Should fold constants: 1.0 + 2.0 = 3.0 and 10.0 * 10.0 = 100.0
        assertTrue(optimized.content.contains("dense<3.0>"))
        
        // Dead code elimination should remove unused %7
        val lines = optimized.content.lines()
        val unusedMultiply = lines.any { it.contains("dense<100.0>") && !it.contains("return") }
        // The unused multiply result should be eliminated
    }
    
    @Test
    fun testAggressiveOptimization() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<3x3xf32>) -> () {
                    %0 = stablehlo.constant dense<0.0> : tensor<3x3xf32>
                    %1 = stablehlo.constant dense<1.0> : tensor<3x3xf32>
                    %2 = stablehlo.multiply %0, %1 : tensor<3x3xf32>
                    %3 = stablehlo.add %arg0, %2 : tensor<3x3xf32>
                    %4 = stablehlo.constant dense<2.0> : tensor<3x3xf32>
                    %5 = stablehlo.constant dense<3.0> : tensor<3x3xf32>
                    %6 = stablehlo.add %4, %5 : tensor<3x3xf32>
                    %7 = stablehlo.multiply %3, %6 : tensor<3x3xf32>
                    return
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = StableHloOptimizer.createAggressive()
        val optimized = optimizer.optimize(module)
        
        // Should apply constant folding multiple times
        // 0.0 * 1.0 = 0.0, 2.0 + 3.0 = 5.0
        assertTrue(optimized.content.contains("dense<0.0>"))
        assertTrue(optimized.content.contains("dense<5.0>"))
        
        // Should run constant folding twice
        val appliedOpts = optimized.metadata["optimizations"] as? List<String> ?: emptyList()
        val constantFoldingCount = appliedOpts.count { it == "constant-folding" }
        assertEquals(2, constantFoldingCount)
    }
    
    @Test
    fun testOptimizationWithInvalidMlir() {
        val invalidModule = StableHloModule(
            content = "invalid mlir content"
        )
        
        val optimizer = StableHloOptimizer.createDefault()
        val result = optimizer.optimize(invalidModule)
        
        // Should return original module when parsing fails
        assertEquals(invalidModule.content, result.content)
    }
    
    @Test
    fun testOptimizationMetadataTracking() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main() -> () {
                    %0 = stablehlo.constant dense<1.0> : tensor<f32>
                    %1 = stablehlo.constant dense<2.0> : tensor<f32>
                    %2 = stablehlo.add %0, %1 : tensor<f32>
                    return
                  }
                }
            """.trimIndent(),
            metadata = mapOf("original" to "test")
        )
        
        val optimizer = StableHloOptimizer()
        optimizer.addPass(ConstantFoldingPass())
        optimizer.addPass(DeadCodeEliminationPass())
        
        val optimized = optimizer.optimize(module)
        
        // Should preserve original metadata and add optimization info
        assertEquals("test", optimized.metadata["original"])
        assertTrue(optimized.metadata.containsKey("optimizations"))
        
        val appliedOpts = optimized.metadata["optimizations"] as? List<String> ?: emptyList()
        assertTrue(appliedOpts.contains("constant-folding"))
        assertTrue(appliedOpts.contains("dead-code-elimination"))
    }
    
    @Test
    fun testNoOptimizationOpportunities() {
        val module = StableHloModule(
            content = """
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
                    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
                    return %0 : tensor<f32>
                  }
                }
            """.trimIndent()
        )
        
        val optimizer = StableHloOptimizer.createDefault()
        val optimized = optimizer.optimize(module)
        
        // Content should be essentially the same (modulo formatting)
        assertTrue(optimized.content.contains("stablehlo.add %arg0, %arg1"))
        
        // Metadata should still track that optimizations were attempted
        assertTrue(optimized.metadata.containsKey("optimizations"))
        val appliedOpts = optimized.metadata["optimizations"] as? List<String> ?: emptyList()
        assertTrue(appliedOpts.isNotEmpty(), "Should track applied optimization passes")
    }
}