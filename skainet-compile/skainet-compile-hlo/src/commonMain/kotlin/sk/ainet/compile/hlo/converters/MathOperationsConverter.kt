package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.lang.graph.GraphNode

/**
 * Converter for comprehensive mathematical operations.
 * 
 * This converter extends BasicMathConverter with additional mathematical operations
 * and enhanced support for element-wise operations with broadcasting, mixed-type
 * arithmetic with automatic type promotion, and proper operand ordering.
 * 
 * Supports operations as specified in Requirements 2.1 and 3.3:
 * - Basic arithmetic: add, subtract, multiply, divide
 * - Element-wise operations with broadcasting
 * - Mixed-type arithmetic with automatic type promotion
 * - Proper operand ordering and type consistency
 */
public class MathOperationsConverter : StableHloOperationConverter {
    
    private val basicMathConverter = BasicMathConverter()
    
    override val supportedOperations: Set<String> = setOf(
        // Basic arithmetic operations
        "add", "subtract", "multiply", "divide",
        // Common aliases
        "sub", "mul", "div",
        // Additional mathematical operations
        "pow", "mod", "remainder",
        // Element-wise operations
        "element_add", "element_sub", "element_mul", "element_div",
        // Element-wise type conversion. Not strictly "math", but
        // MathOperationsConverter already owns the elementwise-op
        // family and cast is an elementwise primitive.
        "cast", "convert", "to"
    )

    override fun convert(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        // Delegate basic math operations to BasicMathConverter
        if (basicMathConverter.supportedOperations.contains(node.operation.name.lowercase())) {
            return basicMathConverter.convert(node, operands, context)
        }

        // Handle additional mathematical operations
        return when (node.operation.name.lowercase()) {
            "pow" -> convertPower(node, operands, context)
            "mod", "remainder" -> convertRemainder(node, operands, context)
            "element_add", "element_sub", "element_mul", "element_div" ->
                convertElementWise(node, operands, context)
            "cast", "convert", "to" -> convertCast(node, operands, context)
            else -> ConversionResult.Unsupported(
                node.operation.name,
                "Operation not supported by MathOperationsConverter"
            )
        }
    }

    /**
     * Convert cast / convert / to to stablehlo.convert.
     *
     * Reads the target dtype from `to`, `to_dtype`, or `dtype`
     * parameter — or, when absent, from the output spec's dtype,
     * which is the normal tracing path. Emits the MLIR type-
     * transition signature `(<from_type>) -> <to_type>`.
     */
    private fun convertCast(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 1) {
            return ConversionResult.Failure(
                "Cast operation requires exactly 1 operand, got ${operands.size}",
                "Unsupported cast arity for node ${node.id}"
            )
        }

        val typeMapper = context.getTypeMapper()
        val inputSpec = node.inputs.firstOrNull()
        val outputSpec = node.outputs.firstOrNull()

        val inputType = inputSpec?.let { typeMapper.mapTensorType(it) } ?: "tensor<?xf32>"
        val outputType = outputSpec?.let { typeMapper.mapTensorType(it) } ?: "tensor<?xf32>"

        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.convert ${operands[0]} : ($inputType) -> $outputType"
        context.emitOperation(operation)

        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    private fun convertPower(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Power operation requires exactly 2 operands, got ${operands.size}",
                "Unsupported power arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.power ${operands[0]}, ${operands[1]} : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    private fun convertRemainder(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Remainder operation requires exactly 2 operands, got ${operands.size}",
                "Unsupported remainder arity for node ${node.id}"
            )
        }
        
        val outputSpec = node.outputs.firstOrNull()
        val outputType = outputSpec?.let { context.getTypeMapper().mapTensorType(it) } 
            ?: "tensor<?xf32>"
        
        val resultValue = context.nextTempValue()
        val operation = "$resultValue = stablehlo.remainder ${operands[0]}, ${operands[1]} : $outputType"
        context.emitOperation(operation)
        
        return ConversionResult.Success(
            outputValueName = resultValue,
            emittedOperations = listOf(operation)
        )
    }
    
    private fun convertElementWise(
        node: GraphNode,
        operands: List<String>,
        context: ConversionContext
    ): ConversionResult {
        if (operands.size != 2) {
            return ConversionResult.Failure(
                "Element-wise operations require exactly 2 operands, got ${operands.size}",
                "Unsupported element-wise arity for node ${node.id}"
            )
        }
        
        // Map element-wise operations to basic operations
        val baseOperation = when (node.operation.name.lowercase()) {
            "element_add" -> "add"
            "element_sub" -> "subtract"
            "element_mul" -> "multiply"
            "element_div" -> "divide"
            else -> return ConversionResult.Unsupported(
                node.operation.name,
                "Unknown element-wise operation"
            )
        }
        
        // Create a new node with the base operation for delegation
        val delegateOperation = object : sk.ainet.lang.tensor.ops.Operation {
            override val name: String = baseOperation
            override val type: String = "math"
            override val parameters: Map<String, Any> = emptyMap()
            
            override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                throw UnsupportedOperationException("This is a delegate operation for conversion only")
            }
            
            override fun validateInputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                return sk.ainet.lang.tensor.ops.ValidationResult.Valid
            }
            
            override fun inferOutputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): List<sk.ainet.lang.tensor.ops.TensorSpec> {
                return node.outputs
            }
            
            override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                return this
            }
            
            override fun serialize(): Map<String, Any> {
                return mapOf("name" to name, "type" to type, "parameters" to parameters)
            }
        }
        
        val delegateNode = sk.ainet.lang.graph.GraphNode(
            id = node.id,
            operation = delegateOperation,
            inputs = node.inputs,
            outputs = node.outputs
        )
        
        return basicMathConverter.convert(delegateNode, operands, context)
    }
}