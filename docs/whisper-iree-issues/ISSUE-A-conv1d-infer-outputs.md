# Conv1d/2d/3dOperation.inferOutputs() echoes input shape instead of computing output shape

## Problem

`Conv1dOperation.inferOutputs()` in `TensorOperations.kt` (line ~439) returns the
input tensor's shape as the output shape, ignoring weight shape, stride, padding,
and dilation:

```kotlin
override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
    require(inputs.size >= 2) { "Conv1d operation requires at least 2 inputs" }
    val outputShape = inputs[0].shape   // <-- BUG: just copies input shape
    return listOf(TensorSpec("conv1d_output", outputShape, inputs[0].dtype, ...))
}
```

Conv2dOperation (line ~471) and Conv3dOperation (line ~503) have the identical bug.

## Expected

```kotlin
override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
    val inShape = inputs[0].shape        // [N, Cin, L]
    val wShape  = inputs[1].shape        // [Cout, Cin/g, K]
    val stride   = (parameters["stride"]   as? Int) ?: 1
    val padding  = (parameters["padding"]  as? Int) ?: 0
    val dilation = (parameters["dilation"] as? Int) ?: 1
    val outShape = if (inShape != null && wShape != null && inShape.size == 3 && wShape.size == 3)
        listOf(inShape[0], wShape[0],
               (inShape[2] + 2*padding - dilation*(wShape[2]-1) - 1)/stride + 1)
    else null
    return listOf(TensorSpec("conv1d_output", outShape, inputs[0].dtype, ...))
}
```

The formula already exists in `VoidTensorOps.calculateConv1dShape()` (line ~747).
`ConvShapeUtils` was added to the JAR but `inferOutputs()` does not call it yet.

## Impact

When the StableHLO converter calls `inferOutputs()` to determine the MLIR output
type, it gets the wrong shape. For Whisper's first conv1d:

```
Input:  [1, 80, 3000]  Weight: [384, 80, 3]  stride=1 padding=1
Actual: [1, 80, 3000]  ← wrong (echoed input)
Expect: [1, 384, 3000] ← correct
```

This produces `tensor<?xf32>` in the MLIR (12 occurrences), which `iree-compile`
rejects.

## Parameters are available

PR #532 stores stride/padding/dilation in `operation.parameters`:

```kotlin
// RecordingExecution.kt:238-261
val params = mapOf("stride" to stride, "padding" to padding, "dilation" to dilation, "groups" to groups)
record(Conv1dOperation<T, V>(params), ...)
```

Verified by test: `assertEquals(1, recorded.operation.parameters["stride"])`

## Suggested fix

Extract `ConvShapeUtils` calls into all three `inferOutputs()` methods.
Single PR covering conv1d/2d/3d since the bug and fix are identical.

## Test

See `Conv1dTapeToHloTest.kt` — asserts `tensor<?` does not appear in output MLIR.
