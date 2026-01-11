# SKaiNET Graph DSL

The SKaiNET Graph DSL provides a way to define complex directed acyclic graphs (DAGs) for machine learning models. Unlike the sequential `nn` DSL, the `dag` DSL allows for arbitrary wiring of nodes, multi-output graphs, and reusable modules.

## Basic Usage

To define a graph, use the `dag` block:

```kotlin
val program = dag {
    val x = input<FP32>("input", TensorSpec("input", listOf(1, 3, 224, 224), "FP32"))
    
    val w = parameter<FP32, Float>("weight") { shape(64, 3, 3, 3) { ones() } }
    val b = constant<FP32, Float>("bias") { shape(64) { zeros() } }
    
    val conv = conv2d(x, w, b, stride = 2 to 2, padding = 1 to 1)
    val activated = relu(conv)
    
    output(activated)
}
```

## Key Concepts

### Inputs, Parameters, and Constants

- `input<T>(name, spec)`: Defines an input node for the graph.
- `parameter<T, V>(name) { ... }`: Defines a learnable parameter node. You can use a builder to specify shape and initialization.
- `constant<T, V>(name) { ... }`: Defines a constant node (e.g., fixed biases or weights).

### Operations

Standard operations like `conv2d`, `relu`, `matmul`, `add`, etc., are available as extension functions within the `DagBuilder` (operations are in sync with TensorOps and implemented extention method via KSP).

### Outputs

A graph can have one or more outputs, defined using the `output()` function.

```kotlin
output(branch1, branch2)
```

## Reusable Modules

You can define reusable graph components using `dagModule`:

```kotlin
val residualBlock = dagModule { inputs ->
    val x = inputs[0]
    val conv1 = conv2d(x, w1, b1, padding = 1 to 1)
    val relu1 = relu(conv1)
    val conv2 = conv2d(relu1, w2, b2, padding = 1 to 1)
    val sum = add(x, conv2)
    listOf(relu(sum))
}

val program = dag {
    val x = input<FP32>("input", spec)
    val out = module(residualBlock, listOf(x))
    output(out[0])
}
```

## Compiling and Validating

Once a `GraphProgram` is built, it can be converted to a `ComputeGraph` for execution or compilation:

```kotlin
val graph = program.toComputeGraph()
val validation = graph.validate()
if (validation is ValidationResult.Valid) {
    // proceed to execution or compilation
}
```

## YOLO-style Example

The Graph DSL is particularly useful for complex architectures like YOLO heads:

```kotlin
val program = dag {
    val input = input<FP32>("input", TensorSpec("input", listOf(1, 3, 640, 640), "FP32"))

    val c1 = conv2d(input, w1, b1, stride = 2 to 2, padding = 1 to 1)
    val c2 = conv2d(c1, w2, b2, stride = 2 to 2, padding = 1 to 1)

    val up = upsample2d(c2, scale = 2 to 2, mode = UpsampleMode.Nearest)

    val head = conv2d(up, wHead, bHead, stride = 1 to 1, padding = 0 to 0)

    output(c2, head) // Multi-scale outputs
}
```
