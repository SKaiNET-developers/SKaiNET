# StableHLO Optimization Framework

The StableHLO optimization framework provides a comprehensive set of optimization passes to improve the performance and efficiency of generated MLIR code. This document describes the available optimizations, their benefits, and how to use them effectively.

## Overview

The optimization framework is built around the `StableHloOptimizer` class, which applies a series of optimization passes to StableHLO MLIR modules. Each pass implements specific optimization strategies that can be combined to achieve maximum performance benefits.

## Available Optimization Passes

### 1. Constant Folding Pass (`ConstantFoldingPass`)

**Purpose**: Evaluates constant expressions at compile time, reducing runtime computation.

**Benefits**:
- Eliminates redundant computations
- Reduces memory traffic
- Enables further optimizations
- Improves cache locality

**Example**:
```mlir
// Before optimization
%0 = stablehlo.constant dense<2.0> : tensor<f32>
%1 = stablehlo.constant dense<3.0> : tensor<f32>
%2 = stablehlo.add %0, %1 : tensor<f32>

// After optimization
%2 = stablehlo.constant dense<5.0> : tensor<f32>
```

**Supported Operations**:
- `stablehlo.add`
- `stablehlo.multiply`
- `stablehlo.subtract`
- `stablehlo.divide`

### 2. Dead Code Elimination Pass (`DeadCodeEliminationPass`)

**Purpose**: Removes operations whose results are never used, reducing code size and complexity.

**Benefits**:
- Reduces memory usage
- Simplifies control flow
- Improves compilation speed
- Eliminates unnecessary computations

**Example**:
```mlir
// Before optimization
%0 = stablehlo.constant dense<1.0> : tensor<f32>
%1 = stablehlo.constant dense<2.0> : tensor<f32>  // Unused
%2 = stablehlo.add %arg0, %0 : tensor<f32>

// After optimization
%0 = stablehlo.constant dense<1.0> : tensor<f32>
%2 = stablehlo.add %arg0, %0 : tensor<f32>
```

### 3. Operation Fusion Pass (`OperationFusionPass`)

**Purpose**: Combines multiple operations into more efficient compound operations.

**Benefits**:
- Reduces memory traffic
- Improves cache locality
- Enables vectorization opportunities
- Reduces kernel launch overhead

**Fusion Patterns**:

#### Add + ReLU Fusion
```mlir
// Before fusion
%0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
%1 = stablehlo.constant dense<0.0> : tensor<2x2xf32>
%2 = stablehlo.maximum %0, %1 : tensor<2x2xf32>

// After fusion
%2 = stablehlo.add %arg0, %arg1 {fused_activation = "relu"} : tensor<2x2xf32>
```

#### Element-wise Operation Fusion
```mlir
// Before fusion
%0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
%1 = stablehlo.multiply %0, %arg2 : tensor<2x2xf32>

// After fusion
%1 = stablehlo.custom %arg0, %arg1, %arg2 {fusion_type = "fused_add_mul"} : tensor<2x2xf32>
```

#### Convolution + Bias Fusion
```mlir
// Before fusion
%0 = stablehlo.convolution(%arg0, %arg1) : tensor<1x32x32x64xf32>
%1 = stablehlo.add %0, %arg2 : tensor<1x32x32x64xf32>

// After fusion
%1 = stablehlo.convolution(%arg0, %arg1, %arg2) {bias = "true"} : tensor<1x32x32x64xf32>
```

## Usage

### Basic Usage

```kotlin
import sk.ainet.compile.hlo.*

// Create a module to optimize
val module = StableHloModule(content = mlirCode)

// Apply default optimizations
val optimizer = StableHloOptimizer.createDefault()
val optimizedModule = optimizer.optimize(module)

println("Optimized MLIR:")
println(optimizedModule.content)
```

### Custom Optimization Pipeline

```kotlin
// Create custom optimizer
val customOptimizer = StableHloOptimizer().apply {
    addPass(ConstantFoldingPass())
    addPass(OperationFusionPass())
    addPass(DeadCodeEliminationPass())
}

val optimizedModule = customOptimizer.optimize(module)
```

### Aggressive Optimization

```kotlin
// Apply aggressive optimizations with multiple passes
val aggressiveOptimizer = StableHloOptimizer.createAggressive()
val optimizedModule = aggressiveOptimizer.optimize(module)
```

## Optimization Strategies

### 1. Default Strategy (`createDefault()`)

Applies a balanced set of optimizations suitable for most use cases:
1. Constant Folding
2. Operation Fusion
3. Dead Code Elimination

### 2. Aggressive Strategy (`createAggressive()`)

Applies more intensive optimizations for maximum performance:
1. Constant Folding (first pass)
2. Operation Fusion
3. Dead Code Elimination
4. Constant Folding (second pass)

### 3. Custom Strategy

Build your own optimization pipeline by adding passes in the desired order:

```kotlin
val optimizer = StableHloOptimizer()
optimizer.addPass(MyCustomPass())
optimizer.addPass(ConstantFoldingPass())
optimizer.addPass(DeadCodeEliminationPass())
```

## Optimization Metadata

The optimization framework tracks applied optimizations in the module metadata:

```kotlin
val optimizedModule = optimizer.optimize(module)
val appliedOptimizations = optimizedModule.metadata["optimizations"] as List<String>
println("Applied optimizations: $appliedOptimizations")
```

## Performance Considerations

### When to Apply Optimizations

1. **Always for Production**: Apply at least default optimizations for production code
2. **Development**: Use lighter optimizations during development for faster iteration
3. **Critical Paths**: Use aggressive optimizations for performance-critical code

### Optimization Order

The order of optimization passes matters:
1. **Constant Folding First**: Enables other optimizations by simplifying expressions
2. **Fusion Before DCE**: Fusion may create new optimization opportunities
3. **DCE Last**: Clean up after other optimizations

### Memory vs. Speed Trade-offs

- **Constant Folding**: Trades compile-time computation for runtime speed
- **Operation Fusion**: Trades code size for execution speed
- **Dead Code Elimination**: Reduces both memory usage and execution time

## Best Practices

### 1. Profile Before Optimizing
```kotlin
val originalOpCount = countOperations(module.content)
val optimizedModule = optimizer.optimize(module)
val optimizedOpCount = countOperations(optimizedModule.content)
val reduction = (originalOpCount - optimizedOpCount) * 100 / originalOpCount
println("Operation count reduced by $reduction%")
```

### 2. Validate Optimized Code
```kotlin
val validator = MlirValidator()
val errors = optimizedModule.validate(validator)
if (errors.isNotEmpty()) {
    println("Validation errors: $errors")
}
```

### 3. Use Appropriate Optimization Level
```kotlin
// For development
val devOptimizer = StableHloOptimizer().apply {
    addPass(ConstantFoldingPass())
}

// For production
val prodOptimizer = StableHloOptimizer.createAggressive()
```

## Examples

See `OptimizationExample.kt` for comprehensive examples demonstrating:
- Basic optimization usage
- Aggressive optimization with multiple passes
- Custom optimization pipelines
- Performance analysis and benefits measurement

## Extending the Framework

### Creating Custom Optimization Passes

```kotlin
class MyCustomPass : OptimizationPass {
    override val name: String = "my-custom-optimization"
    
    override fun apply(module: StableHloModule): StableHloModule {
        // Parse MLIR content
        val parser = MlirParser()
        val parseResult = parser.parse(module.content)
        
        if (parseResult.isFailure) {
            return module // Return original if parsing fails
        }
        
        val structure = parseResult.getOrThrow()
        
        // Apply your optimization logic
        val optimizedOperations = optimizeOperations(structure.operations)
        
        // Reconstruct module
        val optimizedStructure = structure.copy(operations = optimizedOperations)
        val optimizedContent = optimizedStructure.toMlirString()
        
        return module.copy(
            content = optimizedContent,
            metadata = module.metadata + ("optimizations" to 
                (module.metadata["optimizations"] as? List<String> ?: emptyList()) + name)
        )
    }
    
    private fun optimizeOperations(operations: List<ParsedOperation>): List<ParsedOperation> {
        // Your optimization logic here
        return operations
    }
}
```

### Adding New Fusion Patterns

Extend `OperationFusionPass` to recognize new fusion opportunities:

```kotlin
// In tryFuseWithNext method, add new patterns:
if (current.operationType == "stablehlo.my_op" && startIndex + 1 < operations.size) {
    val next = operations[startIndex + 1]
    if (next.operationType == "stablehlo.other_op") {
        return fuseMyPattern(current, next)
    }
}
```

## Troubleshooting

### Common Issues

1. **Parsing Failures**: Ensure MLIR content is well-formed
2. **No Optimizations Applied**: Check that optimization opportunities exist
3. **Incorrect Results**: Validate optimized code with test cases

### Debugging

Enable detailed logging to understand optimization behavior:

```kotlin
val module = optimizer.optimize(inputModule)
println("Applied optimizations: ${module.metadata["optimizations"]}")
```

## Future Enhancements

Planned improvements to the optimization framework:
- Loop optimization passes
- Memory layout optimizations
- Cross-function optimizations
- Hardware-specific optimizations
- Automatic optimization level selection based on profiling data