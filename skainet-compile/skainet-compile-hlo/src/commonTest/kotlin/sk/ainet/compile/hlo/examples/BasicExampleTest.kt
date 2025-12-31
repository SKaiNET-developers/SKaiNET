package sk.ainet.compile.hlo.examples

import kotlin.test.Test
import kotlin.test.assertTrue

class BasicExampleTest {

    @Test
    fun testBasicExample() {
        val mlir = BasicExample.runExample()
        
        println("Generated StableHLO MLIR:")
        println(mlir)
        
        // Verify the generated MLIR contains expected elements
        assertTrue(mlir.contains("module {"))
        assertTrue(mlir.contains("func.func @example_function"))
        assertTrue(mlir.contains("stablehlo.add"))
        assertTrue(mlir.contains("stablehlo.maximum"))
        assertTrue(mlir.contains("tensor<4x4xf32>"))
        assertTrue(mlir.contains("return"))
    }
}