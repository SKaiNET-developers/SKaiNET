package sk.ainet.compile.hlo

import kotlin.test.*

/**
 * Debug test to understand optimization behavior
 */
class OptimizationDebugTest {
    
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
        
        println("Original module:")
        println(module.content)
        
        val optimizer = ConstantFoldingPass()
        val optimized = optimizer.apply(module)
        
        println("\nOptimized module:")
        println(optimized.content)
        
        println("\nMetadata:")
        println(optimized.metadata)
    }
    
    @Test
    fun debugParser() {
        val content = """
            module {
              func.func @main() -> () {
                %0 = stablehlo.constant dense<2.0> : tensor<f32>
                %1 = stablehlo.constant dense<3.0> : tensor<f32>
                %2 = stablehlo.add %0, %1 : tensor<f32>
                return
              }
            }
        """.trimIndent()
        
        val parser = sk.ainet.compile.hlo.validation.MlirParser()
        val result = parser.parse(content)
        
        println("Parse result: ${result.isSuccess}")
        if (result.isSuccess) {
            val structure = result.getOrThrow()
            println("Function name: ${structure.functionName}")
            println("Operations count: ${structure.operations.size}")
            structure.operations.forEach { op ->
                println("  ${op.resultName} = ${op.operationType} ${op.operands.joinToString(", ")} : ${op.resultType}")
            }
        } else {
            println("Parse error: ${result.exceptionOrNull()?.message}")
        }
    }
}