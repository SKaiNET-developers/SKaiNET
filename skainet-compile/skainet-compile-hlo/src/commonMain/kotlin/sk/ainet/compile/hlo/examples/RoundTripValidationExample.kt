package sk.ainet.compile.hlo.examples

import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.compile.hlo.validation.*
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.types.DType

/**
 * Example demonstrating round-trip validation capabilities for StableHLO conversion.
 * 
 * This example shows how to:
 * 1. Generate MLIR from a compute graph
 * 2. Validate the generated MLIR syntax and semantics
 * 3. Perform round-trip validation to ensure correctness
 * 4. Benchmark the conversion performance
 */
public object RoundTripValidationExample {
    
    /**
     * Demonstrate comprehensive validation of StableHLO conversion
     */
    public fun demonstrateValidation(): ValidationDemonstrationResult {
        // Create a simple compute graph
        val graph = createExampleGraph()
        
        // Convert to StableHLO
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "validation_example")
        
        // Perform comprehensive validation
        val testUtilities = MlirTestUtilities()
        val correctnessResult = testUtilities.verifyCorrectness(module)
        
        // Perform round-trip validation
        val roundTripValidator = RoundTripValidator()
        val roundTripResult = roundTripValidator.validateRoundTrip(module)
        
        // Benchmark the conversion
        val benchmark = PerformanceBenchmark()
        val performanceMetrics = benchmark.benchmarkConversion(converter, graph)
        
        return ValidationDemonstrationResult(
            generatedModule = module,
            correctnessResult = correctnessResult,
            roundTripResult = roundTripResult,
            performanceMetrics = performanceMetrics
        )
    }
    
    /**
     * Demonstrate performance benchmarking
     */
    public fun demonstratePerformanceBenchmarking(): BenchmarkResults {
        val graph = createExampleGraph()
        val converter = StableHloConverterFactory.createBasic()
        val benchmark = PerformanceBenchmark()
        
        return benchmark.benchmarkMultipleRuns(
            converter = converter,
            graph = graph,
            runs = 5,
            testName = "Round-trip Validation Example"
        )
    }
    
    /**
     * Demonstrate MLIR parsing capabilities
     */
    public fun demonstrateParsing(): ParsingDemonstrationResult {
        val mlirContent = """
            module {
              func.func @example(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> () {
                %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
                %1 = stablehlo.constant dense<0.0> : tensor<2x3xf32>
                %2 = stablehlo.maximum %0, %1 : tensor<2x3xf32>
                return
              }
            }
        """.trimIndent()
        
        val parser = MlirParser()
        val parseResult = parser.parse(mlirContent)
        
        return if (parseResult.isSuccess) {
            val structure = parseResult.getOrThrow()
            val reconstructed = structure.toMlirString()
            
            ParsingDemonstrationResult(
                originalContent = mlirContent,
                parsedStructure = structure,
                reconstructedContent = reconstructed,
                parseSuccess = true,
                errors = emptyList()
            )
        } else {
            ParsingDemonstrationResult(
                originalContent = mlirContent,
                parsedStructure = null,
                reconstructedContent = "",
                parseSuccess = false,
                errors = listOf(parseResult.exceptionOrNull()?.message ?: "Unknown parse error")
            )
        }
    }
    
    private fun createExampleGraph(): DefaultComputeGraph {
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
}

/**
 * Result of validation demonstration
 */
public data class ValidationDemonstrationResult(
    val generatedModule: sk.ainet.compile.hlo.StableHloModule,
    val correctnessResult: CorrectnessVerificationResult,
    val roundTripResult: RoundTripValidationResult,
    val performanceMetrics: ConversionMetrics
) {
    public fun summary(): String {
        return buildString {
            appendLine("=== Round-trip Validation Demonstration ===")
            appendLine()
            appendLine("Generated MLIR Module:")
            appendLine("  Function: ${generatedModule.functionName}")
            appendLine("  Size: ${generatedModule.content.length} characters")
            appendLine()
            appendLine("Correctness Validation:")
            appendLine("  Overall: ${if (correctnessResult.isCorrect) "PASS" else "FAIL"}")
            appendLine("  Syntax: ${if (correctnessResult.syntaxValid) "PASS" else "FAIL"}")
            appendLine("  Semantics: ${if (correctnessResult.semanticValid) "PASS" else "FAIL"}")
            appendLine("  Round-trip: ${if (correctnessResult.roundTripValid) "PASS" else "FAIL"}")
            if (correctnessResult.errors.isNotEmpty()) {
                appendLine("  Errors: ${correctnessResult.errors.joinToString(", ")}")
            }
            appendLine()
            appendLine("Round-trip Validation:")
            when (roundTripResult) {
                is RoundTripValidationResult.Success -> {
                    appendLine("  Status: SUCCESS")
                    appendLine("  Equivalent: ${roundTripResult.equivalenceReport.isEquivalent}")
                }
                is RoundTripValidationResult.Failure -> {
                    appendLine("  Status: FAILURE")
                    appendLine("  Stage: ${roundTripResult.stage}")
                    appendLine("  Errors: ${roundTripResult.errors.joinToString(", ")}")
                }
            }
            appendLine()
            appendLine("Performance Metrics:")
            appendLine("  Conversion time: ${performanceMetrics.conversionTime}")
            appendLine("  Validation time: ${performanceMetrics.validationTime}")
            appendLine("  Total time: ${performanceMetrics.totalTime}")
            appendLine("  Module size: ${performanceMetrics.moduleSize} bytes")
            appendLine("  Operations: ${performanceMetrics.operationCount}")
        }
    }
}

/**
 * Result of parsing demonstration
 */
public data class ParsingDemonstrationResult(
    val originalContent: String,
    val parsedStructure: ParsedMlirStructure?,
    val reconstructedContent: String,
    val parseSuccess: Boolean,
    val errors: List<String>
) {
    public fun summary(): String {
        return buildString {
            appendLine("=== MLIR Parsing Demonstration ===")
            appendLine()
            appendLine("Parse Status: ${if (parseSuccess) "SUCCESS" else "FAILURE"}")
            
            if (parseSuccess && parsedStructure != null) {
                appendLine("Function: ${parsedStructure.functionName}")
                appendLine("Parameters: ${parsedStructure.functionSignature.parameters.size}")
                appendLine("Operations: ${parsedStructure.operations.size}")
                appendLine()
                appendLine("Round-trip successful: ${reconstructedContent.isNotEmpty()}")
            } else {
                appendLine("Errors: ${errors.joinToString(", ")}")
            }
        }
    }
}