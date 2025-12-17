# StableHLO Backend Completion Design Document

## Overview

This design document outlines the completion of the StableHLO backend for SKaiNET, which will enable compilation of computational graphs to StableHLO MLIR format. The current implementation provides basic functionality for add, matmul, and relu operations but needs to be expanded to support the full range of SKaiNET operations including mathematical operations, neural network layers, shape manipulations, and activation functions.

The StableHLO backend will serve as a bridge between SKaiNET's high-level DSL and low-level optimized execution through XLA and other MLIR-based compilation toolchains.

## Architecture

### Current Architecture

The existing implementation consists of:
- `StableHloModule` data class for representing MLIR output
- `toStableHlo()` function for converting ComputeGraph to MLIR
- Basic operation mapping for add, matmul, and relu
- Simple SSA value management and type mapping

### Enhanced Architecture

The enhanced architecture will include:

1. **Operation Registry System**: A pluggable system for registering StableHLO operation converters
2. **Type System Integration**: Comprehensive mapping between SKaiNET and MLIR type systems
3. **Optimization Framework**: Infrastructure for applying StableHLO-specific optimizations
4. **Validation System**: MLIR syntax and semantic validation
5. **Error Handling**: Comprehensive error reporting and fallback mechanisms

### Component Hierarchy

```
skainet-compile-hlo/
├── core/
│   ├── StableHloModule.kt (existing)
│   ├── StableHloConverter.kt (new)
│   └── StableHloRegistry.kt (new)
├── operations/
│   ├── MathOperations.kt (new)
│   ├── LinalgOperations.kt (new)
│   ├── NeuralNetOperations.kt (new)
│   ├── ShapeOperations.kt (new)
│   └── ActivationOperations.kt (new)
├── types/
│   ├── TypeMapper.kt (new)
│   └── ShapeInference.kt (new)
├── validation/
│   └── MlirValidator.kt (new)
└── optimization/
    └── StableHloOptimizer.kt (new)
```

## Components and Interfaces

### StableHloConverter

The main converter class that orchestrates the conversion process:

```kotlin
public class StableHloConverter(
    private val registry: StableHloOperationRegistry,
    private val typeMapper: TypeMapper,
    private val validator: MlirValidator? = null
) {
    public fun convert(graph: ComputeGraph, functionName: String = "main"): StableHloModule
    public fun convertWithOptimization(graph: ComputeGraph, optimizer: StableHloOptimizer): StableHloModule
}
```

### StableHloOperationRegistry

A registry system for operation converters:

```kotlin
public interface StableHloOperationConverter {
    public val supportedOperations: Set<String>
    public fun convert(
        node: GraphNode, 
        operands: List<String>, 
        context: ConversionContext
    ): ConversionResult
}

public class StableHloOperationRegistry {
    public fun register(converter: StableHloOperationConverter)
    public fun getConverter(operationName: String): StableHloOperationConverter?
    public fun getSupportedOperations(): Set<String>
}
```

### TypeMapper

Handles type system mapping between SKaiNET and MLIR:

```kotlin
public class TypeMapper {
    public fun mapDType(dtype: String): String
    public fun mapTensorType(spec: TensorSpec): String
    public fun mapFunctionSignature(inputs: List<TensorSpec>, outputs: List<TensorSpec>): String
    public fun inferBroadcastType(left: TensorSpec, right: TensorSpec): TensorSpec
}
```

### ConversionContext

Context object for maintaining state during conversion:

```kotlin
public class ConversionContext(
    private val valueNames: MutableMap<String, String>,
    private val typeMapper: TypeMapper,
    private val stringBuilder: StringBuilder
) {
    public fun getValueName(nodeId: String): String?
    public fun setValueName(nodeId: String, valueName: String)
    public fun nextTempValue(): String
    public fun emitOperation(operation: String)
    public fun emitComment(comment: String)
}
```

## Data Models

### StableHloModule (Enhanced)

```kotlin
public data class StableHloModule(
    val content: String,
    val functionName: String,
    val inputSpecs: List<TensorSpec>,
    val outputSpecs: List<TensorSpec>,
    val metadata: Map<String, Any> = emptyMap()
) {
    public fun validate(): ValidationResult
    public fun optimize(optimizer: StableHloOptimizer): StableHloModule
}
```

### ConversionResult

```kotlin
public sealed class ConversionResult {
    public data class Success(
        val outputValueName: String,
        val emittedOperations: List<String>
    ) : ConversionResult()
    
    public data class Failure(
        val error: String,
        val fallbackComment: String? = null
    ) : ConversionResult()
    
    public data class Unsupported(
        val operationName: String,
        val reason: String
    ) : ConversionResult()
}
```

### OperationMapping

```kotlin
public data class OperationMapping(
    val skainetOperation: String,
    val stableHloOperation: String,
    val attributeMapping: Map<String, String> = emptyMap(),
    val customConverter: ((GraphNode, List<String>, ConversionContext) -> ConversionResult)? = null
)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After reviewing all properties identified in the prework, several redundancies were identified:
- Properties 1.2 and 5.1 both address MLIR syntax validation - consolidated into Property 1
- Properties about infrastructure (4.1, 4.3, 4.4, 5.4, 5.5, 6.3, 6.4, 6.5) are design concerns rather than testable behaviors
- Properties can be grouped by functional area for better organization

### Correctness Properties

Property 1: Valid MLIR Generation
*For any* ComputeGraph containing supported operations, the generated StableHLO MLIR code should be syntactically valid and parseable by MLIR tools
**Validates: Requirements 1.1, 1.2, 5.1**

Property 2: SSA Form Preservation
*For any* sequence of chained operations in a ComputeGraph, the generated MLIR should maintain proper SSA form with correct value dependencies
**Validates: Requirements 1.3**

Property 3: Type Mapping Correctness
*For any* TensorSpec with specified shape and data type, the emitted MLIR type annotations should correctly represent the SKaiNET types using appropriate MLIR syntax
**Validates: Requirements 1.4, 3.1, 3.5**

Property 4: Error Handling Consistency
*For any* unsupported operation in a ComputeGraph, the system should emit clear error messages or fallback comments without crashing
**Validates: Requirements 1.5, 5.3**

Property 5: Mathematical Operation Mapping
*For any* mathematical operation (add, subtract, multiply, divide), the generated StableHLO should contain the corresponding arithmetic operation with correct operands
**Validates: Requirements 2.1**

Property 6: Linear Algebra Operation Mapping
*For any* linear algebra operation (matmul, transpose), the generated StableHLO should contain the appropriate stablehlo.dot_general or stablehlo.transpose operation
**Validates: Requirements 2.2**

Property 7: Activation Function Implementation
*For any* activation function (relu, sigmoid, softmax), the generated StableHLO should implement the function using appropriate primitive operations
**Validates: Requirements 2.3**

Property 8: Convolutional Operation Mapping
*For any* conv2d operation, the generated StableHLO should contain stablehlo.convolution with properly mapped attributes
**Validates: Requirements 2.4**

Property 9: Shape Operation Mapping
*For any* shape operation (reshape, flatten, squeeze, unsqueeze), the generated StableHLO should contain appropriate stablehlo.reshape or stablehlo.broadcast_in_dim operations
**Validates: Requirements 2.5**

Property 10: Dynamic Shape Handling
*For any* tensor with dynamic dimensions, the generated MLIR should use correct dynamic dimension syntax (? in tensor types)
**Validates: Requirements 3.2**

Property 11: Broadcasting Operation Generation
*For any* operation requiring broadcasting between tensors of different shapes, the system should emit explicit broadcast operations in StableHLO
**Validates: Requirements 3.3**

Property 12: Type Conversion Handling
*For any* operation involving mixed data types, the system should emit appropriate type conversion operations in StableHLO
**Validates: Requirements 3.4**

Property 13: Constant Value Handling
*For any* constant value in the computation graph, the generated StableHLO should contain stablehlo.constant operations with correct attributes
**Validates: Requirements 4.2**

Property 14: Control Flow Mapping
*For any* control flow construct in the graph, the system should emit appropriate stablehlo.if or stablehlo.while operations
**Validates: Requirements 4.5**

Property 15: Round-trip Validation
*For any* generated StableHLO module, parsing and re-processing should preserve the essential computational semantics
**Validates: Requirements 5.2**

Property 16: ComputeGraph Interface Compatibility
*For any* valid ComputeGraph implementation, the StableHLO backend should successfully process it without interface violations
**Validates: Requirements 6.1**

Property 17: Operation Metadata Integration
*For any* operation with KSP-generated metadata, the StableHLO backend should correctly utilize the metadata for conversion
**Validates: Requirements 6.2**

## Error Handling

The error handling strategy will include:

1. **Validation Errors**: Pre-conversion validation of graph structure and operation compatibility
2. **Conversion Errors**: Graceful handling of unsupported operations with fallback comments
3. **Type Errors**: Clear error messages for type incompatibilities
4. **MLIR Syntax Errors**: Post-generation validation with detailed error reporting

### Error Categories

```kotlin
public sealed class StableHloError {
    public data class UnsupportedOperation(
        val operationName: String,
        val nodeId: String,
        val reason: String
    ) : StableHloError()
    
    public data class TypeMappingError(
        val sourceType: String,
        val context: String
    ) : StableHloError()
    
    public data class GraphValidationError(
        val message: String,
        val nodeId: String? = null
    ) : StableHloError()
    
    public data class MlirSyntaxError(
        val line: Int,
        val column: Int,
        val message: String
    ) : StableHloError()
}
```

## Testing Strategy

### Unit Testing Approach

Unit tests will focus on:
- Individual operation converters with known inputs and expected outputs
- Type mapping functions with various SKaiNET types
- Error handling with invalid inputs
- MLIR syntax validation with malformed inputs

### Property-Based Testing Approach

Property-based tests will use **Kotest Property Testing** framework and will verify:
- Universal properties across all supported operations
- Type system correctness across random tensor specifications
- SSA form preservation across random graph structures
- Error handling consistency across various failure scenarios

Each property-based test will run a minimum of 100 iterations to ensure comprehensive coverage of the input space.

### Integration Testing

Integration tests will verify:
- End-to-end conversion of complete neural network models
- Compatibility with existing SKaiNET execution contexts
- Performance characteristics of generated StableHLO code

### Test Data Generation

Smart generators will be implemented for:
- Random but valid ComputeGraph instances
- Tensor specifications with realistic shapes and types
- Operation sequences that form valid neural network patterns
- Edge cases like empty graphs, single-node graphs, and deeply nested structures

## Implementation Plan

The implementation will follow an incremental approach:

1. **Core Infrastructure**: Implement the converter framework and registry system
2. **Basic Operations**: Extend support for all mathematical and linear algebra operations
3. **Neural Network Operations**: Add support for convolutions, pooling, and activations
4. **Shape Operations**: Implement reshape, broadcast, and dimension manipulation operations
5. **Advanced Features**: Add optimization passes and validation systems
6. **Integration**: Ensure compatibility with existing SKaiNET infrastructure

Each phase will include comprehensive testing and validation before proceeding to the next phase.