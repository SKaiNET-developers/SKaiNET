package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import sk.ainet.compile.hlo.validation.*

class MlirParserTest {
    
    private val parser = MlirParser()
    
    @Test
    fun parse_simpleModule_succeeds() {
        val mlirContent = """
            module {
              func.func @main(%arg0: tensor<2x3xf32>) -> () {
                %0 = stablehlo.relu %arg0 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        
        assertTrue(result.isSuccess, "Parsing should succeed")
        val structure = result.getOrThrow()
        
        assertEquals("main", structure.functionName)
        assertEquals(1, structure.functionSignature.parameters.size)
        assertEquals("%arg0", structure.functionSignature.parameters[0].name)
        assertEquals("tensor<2x3xf32>", structure.functionSignature.parameters[0].type)
        assertEquals(1, structure.operations.size)
        assertEquals("stablehlo.relu", structure.operations[0].operationType)
    }
    
    @Test
    fun parse_multipleOperations_extractsAllOperations() {
        val mlirContent = """
            module {
              func.func @test(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> () {
                %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
                %1 = stablehlo.constant dense<0.0> : tensor<2x3xf32>
                %2 = stablehlo.maximum %0, %1 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        
        assertTrue(result.isSuccess, "Parsing should succeed")
        val structure = result.getOrThrow()
        
        assertEquals("test", structure.functionName)
        assertEquals(2, structure.functionSignature.parameters.size)
        assertEquals(3, structure.operations.size)
        
        // Check operation types
        assertEquals("stablehlo.add", structure.operations[0].operationType)
        assertEquals("stablehlo.constant", structure.operations[1].operationType)
        assertEquals("stablehlo.maximum", structure.operations[2].operationType)
        
        // Check operands
        assertEquals(listOf("%arg0", "%arg1"), structure.operations[0].operands)
        assertEquals(listOf("%0", "%1"), structure.operations[2].operands)
    }
    
    @Test
    fun parse_emptyContent_fails() {
        val result = parser.parse("")
        
        assertTrue(result.isFailure, "Parsing empty content should fail")
    }
    
    @Test
    fun parse_invalidContent_fails() {
        val invalidContent = """
            this is not valid mlir
            no structure at all
        """.trimIndent()
        
        val result = parser.parse(invalidContent)
        
        assertTrue(result.isFailure, "Parsing invalid content should fail")
    }
    
    @Test
    fun parse_noFunctionDeclaration_fails() {
        val mlirContent = """
            module {
              // No function declaration
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        
        assertTrue(result.isFailure, "Parsing without function declaration should fail")
    }
    
    @Test
    fun parsedStructure_toMlirString_producesValidMlir() {
        val originalContent = """
            module {
              func.func @main(%arg0: tensor<2x3xf32>) -> () {
                %0 = stablehlo.relu %arg0 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val parseResult = parser.parse(originalContent)
        assertTrue(parseResult.isSuccess, "Initial parsing should succeed")
        
        val structure = parseResult.getOrThrow()
        val reconstructed = structure.toMlirString()
        
        // The reconstructed MLIR should be parseable
        val reparseResult = parser.parse(reconstructed)
        assertTrue(reparseResult.isSuccess, "Reconstructed MLIR should be parseable")
        
        val reStructure = reparseResult.getOrThrow()
        assertEquals(structure.functionName, reStructure.functionName)
        assertEquals(structure.operations.size, reStructure.operations.size)
    }
    
    @Test
    fun validateParsability_validContent_returnsNoErrors() {
        val validContent = """
            module {
              func.func @main(%arg0: tensor<2x3xf32>) -> () {
                %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val errors = parser.validateParsability(validContent)
        
        assertTrue(errors.isEmpty(), "Valid content should have no parsability errors")
    }
    
    @Test
    fun validateParsability_invalidContent_returnsErrors() {
        val invalidContent = """
            invalid content
            not mlir at all
        """.trimIndent()
        
        val errors = parser.validateParsability(invalidContent)
        
        assertTrue(errors.isNotEmpty(), "Invalid content should have parsability errors")
    }
    
    @Test
    fun functionSignature_multipleParameters_parsedCorrectly() {
        val mlirContent = """
            module {
              func.func @test(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf64>, %arg2: tensor<1xi32>) -> () {
                return
              }
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        assertTrue(result.isSuccess, "Parsing should succeed")
        
        val structure = result.getOrThrow()
        val signature = structure.functionSignature
        
        assertEquals(3, signature.parameters.size)
        assertEquals("%arg0", signature.parameters[0].name)
        assertEquals("tensor<2x3xf32>", signature.parameters[0].type)
        assertEquals("%arg1", signature.parameters[1].name)
        assertEquals("tensor<3x4xf64>", signature.parameters[1].type)
        assertEquals("%arg2", signature.parameters[2].name)
        assertEquals("tensor<1xi32>", signature.parameters[2].type)
    }
    
    @Test
    fun functionSignature_withReturnTypes_parsedCorrectly() {
        val mlirContent = """
            module {
              func.func @test(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>) {
                return %arg0 : tensor<2x3xf32>
              }
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        assertTrue(result.isSuccess, "Parsing should succeed")
        
        val structure = result.getOrThrow()
        val signature = structure.functionSignature
        
        assertEquals(1, signature.returnTypes.size)
        assertEquals("tensor<2x3xf32>", signature.returnTypes[0])
    }
    
    @Test
    fun operations_withComplexTypes_parsedCorrectly() {
        val mlirContent = """
            module {
              func.func @test(%arg0: tensor<?x?xf32>) -> () {
                %0 = stablehlo.reshape %arg0 : (tensor<?x?xf32>) -> tensor<?xf32>
                return
              }
            }
        """.trimIndent()
        
        val result = parser.parse(mlirContent)
        assertTrue(result.isSuccess, "Parsing should succeed")
        
        val structure = result.getOrThrow()
        assertEquals(1, structure.operations.size)
        
        val operation = structure.operations[0]
        assertEquals("stablehlo.reshape", operation.operationType)
        assertEquals(listOf("%arg0"), operation.operands)
        assertNotNull(operation.resultType)
    }
}