# Skainet Autograd: From Math to Optimization

This article explores how Skainet handles automatic differentiation, graph conversion, and optimization using a practical example: **Cosine Distance**.

## 1. The Core Expression
We start with two vectors and calculate their cosine distance.
Cosine distance is defined as:
`1 - (a · b) / (||a|| * ||b||)`

In Skainet, this is expressed simply using tensor operations:
```kotlin
val distance = aTensor.cosineDistance(bTensor)
```

## 2. Conversion to Graph
When operations are executed within a `record` block on a `GraphExecutionContext`, Skainet captures the execution trace and can convert it into a `ComputeGraph`.

### Untrained Forward Graph
Before any training happens, the forward pass defines the structure of our computation.
- **Nodes**: Operations like `multiply`, `sum`, `sqrt`, `divide`.
- **Edges**: Data flow between operations.

Visualization (DOT format):
```dot
digraph {
    rankdir=LR;
    n0_multiply [label="multiply"];
    n1_sum [label="sum"];
    ...
    n10_divide -> n11_rsubScalar;
}
```

## 3. Loss Function
In our case, we want to minimize the cosine distance itself, so the `distance` tensor acts as our `loss`.
```kotlin
val loss = distance
```

## 4. Graph Inversion (Autograd)
Skainet's `DefaultGradientTape` performs the backward pass. It traverses the recorded operations in reverse order, applying adjoint rules (chain rule) to compute gradients.

```kotlin
tape.computeGradients(targets = listOf(loss), sources = listOf(aTensor))
```
This populates `aTensor.grad` with the sensitivity of the loss with respect to each element in `a`.

## 5. Optimization Step
Finally, we use an `Optimizer` (like SGD) to update our parameters based on the computed gradients.
```kotlin
val optimizer = sgd(lr = 0.5)
optimizer.addParameter(aParam)

optimizer.step()
optimizer.zeroGrad()
```

After the step, the distance between the vectors decreases (or similarity increases), demonstrating the successful training loop.

## Conclusion
Skainet Autograd provides a transparent way to move from high-level math expressions to optimized execution graphs and trainable models.
