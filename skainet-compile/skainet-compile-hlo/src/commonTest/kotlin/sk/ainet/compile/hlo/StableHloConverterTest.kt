package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import kotlin.test.assertEquals
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.types.DType
import sk.ainet.compile.hlo.converters.LegacyOperationsConverter

class StableHloConverterTest {

    @Test
    fun testConverterFactory() {
        val converter = StableHloConverterFactory.createBasic()
        assertNotNull(converter)
    }

    @Test
    fun testBasicConversion() {
        // Create a simple graph with two inputs and an add operation
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(2, 3), "FP32"))
        )
        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)

        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_function")

        // Verify the module structure
        assertEquals("test_function", module.functionName)
        assertEquals(2, module.inputSpecs.size)
        assertTrue(module.content.contains("module {"))
        assertTrue(module.content.contains("func.func @test_function"))
        assertTrue(module.content.contains("stablehlo.add"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testTypeMapper() {
        val typeMapper = TypeMapper()
        
        // Test basic type mapping
        assertEquals("f32", typeMapper.mapDType("FP32"))
        assertEquals("f64", typeMapper.mapDType("FP64"))
        assertEquals("i32", typeMapper.mapDType("I32"))
        
        // Test tensor type mapping
        val spec = TensorSpec("test", listOf(2, 3), "FP32")
        assertEquals("tensor<2x3xf32>", typeMapper.mapTensorType(spec))
        
        // Test dynamic shape
        val dynamicSpec = TensorSpec("test", null, "FP32")
        assertEquals("tensor<?xf32>", typeMapper.mapTensorType(dynamicSpec))
    }

    @Test
    fun testOperationRegistry() {
        val registry = StableHloOperationRegistry()
        val converter = LegacyOperationsConverter()
        
        registry.register(converter)
        
        assertTrue(registry.isSupported("add"))
        assertTrue(registry.isSupported("matmul"))
        assertTrue(registry.isSupported("relu"))
        
        val stats = registry.getStats()
        assertEquals(1, stats.converterCount)
        assertEquals(3, stats.operationCount)
    }

    @Test
    fun testMlirValidator() {
        val validator = MlirValidator()
        
        // Test valid MLIR
        val validMlir = """
            module {
              func.func @main(%arg0: tensor<2x3xf32>) -> () {
                %0 = stablehlo.add %arg0, %arg0 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val errors = validator.validate(validMlir)
        assertTrue(errors.isEmpty(), "Valid MLIR should not have errors: $errors")
        
        // Test invalid MLIR (unbalanced braces)
        val invalidMlir = """
            module {
              func.func @main() -> () {
                return
              }
            // Missing closing brace
        """.trimIndent()
        
        val invalidErrors = validator.validate(invalidMlir)
        assertTrue(invalidErrors.isNotEmpty(), "Invalid MLIR should have errors")
    }
}