package sk.ainet.compile.hlo.validation

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.MlirValidator
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.types.DType

/**
 * Result of MLIR correctness verification
 */
public data class CorrectnessVerificationResult(
    val isCorrect: Boolean,
    val syntaxValid: Boolean,
    val semanticValid: Boolean,
    val roundTripValid: Boolean,
    val errors: List<String> = emptyList(),
    val warnings: List<String> = emptyList()
)

/**
 * Test utilities for verifying generated MLIR correctness.
 * 
 * This class provides comprehensive testing utilities for validating
 * StableHLO MLIR generation, including syntax validation, semantic checks,
 * and round-trip validation.
 */
public class MlirTestUtilities {
    
    private val mlirValidator = MlirValidator()
    private val roundTripValidator = RoundTripValidator()
    
    /**
     * Perform comprehensive correctness verification of a StableHLO module
     */
    public fun verifyCorrectness(module: StableHloModule): CorrectnessVerificationResult {
        val errors = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        
        // 1. Syntax validation
        val syntaxErrors = mlirValidator.validate(module.content)
        val syntaxValid = syntaxErrors.isEmpty()
        if (!syntaxValid) {
            errors.addAll(syntaxErrors.map { "Syntax: $it" })
        }
        
        // 2. Semantic validation
        val semanticErrors = mlirValidator.validateModule(module.content)
        val semanticValid = semanticErrors.isEmpty()
        if (!semanticValid) {
            errors.addAll(semanticErrors.map { "Semantic: $it" })
        }
        
        // 3. Round-trip validation
        val roundTripResult = roundTripValidator.validateRoundTrip(module)
        val roundTripValid = roundTripResult is RoundTripValidationResult.Success
        if (!roundTripValid && roundTripResult is RoundTripValidationResult.Failure) {
            errors.addAll(roundTripResult.errors.map { "RoundTrip: $it" })
        }
        
        val isCorrect = syntaxValid && semanticValid && roundTripValid
        
        return CorrectnessVerificationResult(
            isCorrect = isCorrect,
            syntaxValid = syntaxValid,
            semanticValid = semanticValid,
            roundTripValid = roundTripValid,
            errors = errors,
            warnings = warnings
        )
    }
    
    /**
     * Verify that a module contains expected operations
     */
    public fun verifyOperations(
        module: StableHloModule,
        expectedOperations: List<String>
    ): List<String> {
        val errors = mutableListOf<String>()
        val content = module.content
        
        for (expectedOp in expectedOperations) {
            if (!content.contains("stablehlo.$expectedOp")) {
                errors.add("Missing expected operation: stablehlo.$expectedOp")
            }
        }
        
        return errors
    }
    
    /**
     * Verify that a module has correct SSA form
     */
    public fun verifySSAForm(module: StableHloModule): List<String> {
        val errors = mutableListOf<String>()
        val lines = module.content.lines()
        val definedValues = mutableSetOf<String>()
        val usedValues = mutableSetOf<String>()
        
        for ((lineNum, line) in lines.withIndex()) {
            val trimmed = line.trim()
            
            // Skip empty lines and comments
            if (trimmed.isEmpty() || trimmed.startsWith("//")) continue
            
            // Extract defined SSA values
            if (trimmed.contains(" = ")) {
                val parts = trimmed.split(" = ", limit = 2)
                if (parts.size == 2) {
                    val valueName = parts[0].trim()
                    if (valueName.startsWith("%")) {
                        if (definedValues.contains(valueName)) {
                            errors.add("Line ${lineNum + 1}: SSA value $valueName redefined")
                        }
                        definedValues.add(valueName)
                    }
                }
            }
            
            // Extract used SSA values
            val regex = Regex("""%[a-zA-Z0-9_]+""")
            regex.findAll(trimmed).forEach { match ->
                usedValues.add(match.value)
            }
        }
        
        // Check for undefined values (excluding function arguments)
        for (used in usedValues) {
            if (!used.startsWith("%arg") && !definedValues.contains(used)) {
                errors.add("Undefined SSA value: $used")
            }
        }
        
        return errors
    }
    
    /**
     * Verify that a module has correct type annotations
     */
    public fun verifyTypeAnnotations(module: StableHloModule): List<String> {
        val errors = mutableListOf<String>()
        val lines = module.content.lines()
        
        for ((lineNum, line) in lines.withIndex()) {
            val trimmed = line.trim()
            
            // Check operations have type annotations
            if (trimmed.contains("stablehlo.") && trimmed.contains(" = ")) {
                if (!trimmed.contains(" : ")) {
                    errors.add("Line ${lineNum + 1}: Missing type annotation for operation")
                }
            }
            
            // Validate tensor type format
            val tensorRegex = Regex("""tensor<[^>]+>""")
            tensorRegex.findAll(trimmed).forEach { match ->
                if (!isValidTensorType(match.value)) {
                    errors.add("Line ${lineNum + 1}: Invalid tensor type format: ${match.value}")
                }
            }
        }
        
        return errors
    }
    
    /**
     * Create a simple test graph for validation testing
     */
    public fun createSimpleTestGraph(): ComputeGraph {
        val graph = DefaultComputeGraph()
        
        val inputA = GraphNode(
            id = "input_a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        
        val inputB = GraphNode(
            id = "input_b", 
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(2, 3), "FP32"))
        )
        
        val add = GraphNode(
            id = "add_op",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )
        
        val relu = GraphNode(
            id = "relu_op",
            operation = ReluOperation<DType, Any>(),
            inputs = listOf(TensorSpec("c", listOf(2, 3), "FP32")),
            outputs = listOf(TensorSpec("d", listOf(2, 3), "FP32"))
        )
        
        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)
        graph.addNode(relu)
        
        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))
        graph.addEdge(GraphEdge("e3", add, relu, 0, 0, add.outputs[0]))
        
        return graph
    }
    
    /**
     * Create a complex test graph with multiple operation types
     */
    public fun createComplexTestGraph(): ComputeGraph {
        // This would create a more complex graph with various operations
        // For now, return the simple graph as a placeholder
        return createSimpleTestGraph()
    }
    
    /**
     * Validate that generated MLIR matches expected patterns
     */
    public fun validateMlirPatterns(
        module: StableHloModule,
        expectedPatterns: List<String>
    ): List<String> {
        val errors = mutableListOf<String>()
        val content = module.content
        
        for (pattern in expectedPatterns) {
            if (!content.contains(pattern)) {
                errors.add("Missing expected pattern: $pattern")
            }
        }
        
        return errors
    }
    
    /**
     * Extract performance metrics from module generation
     */
    public fun extractPerformanceMetrics(module: StableHloModule): Map<String, Any> {
        val metrics = mutableMapOf<String, Any>()
        
        // Basic metrics
        metrics["content_length"] = module.content.length
        metrics["line_count"] = module.content.lines().size
        metrics["operation_count"] = countOperations(module.content)
        metrics["ssa_value_count"] = countSSAValues(module.content)
        
        return metrics
    }
    
    private fun isValidTensorType(tensorType: String): Boolean {
        // Basic validation of tensor type format: tensor<shape x element_type>
        val regex = Regex("""tensor<[^>]*x(f32|f64|i32|i64)>""")
        return regex.matches(tensorType) || tensorType == "tensor<?xf32>" // Allow dynamic shapes
    }
    
    private fun countOperations(content: String): Int {
        return content.lines().count { line ->
            line.trim().contains("stablehlo.")
        }
    }
    
    private fun countSSAValues(content: String): Int {
        val regex = Regex("""%[a-zA-Z0-9_]+""")
        return regex.findAll(content).count()
    }
}

/**
 * Builder for creating test scenarios
 */
public class TestScenarioBuilder {
    private val scenarios = mutableListOf<TestScenario>()
    
    public fun addScenario(
        name: String,
        graph: ComputeGraph,
        expectedOperations: List<String> = emptyList(),
        expectedPatterns: List<String> = emptyList()
    ): TestScenarioBuilder {
        scenarios.add(TestScenario(name, graph, expectedOperations, expectedPatterns))
        return this
    }
    
    public fun build(): List<TestScenario> = scenarios.toList()
}

/**
 * Represents a test scenario for MLIR validation
 */
public data class TestScenario(
    val name: String,
    val graph: ComputeGraph,
    val expectedOperations: List<String> = emptyList(),
    val expectedPatterns: List<String> = emptyList()
)