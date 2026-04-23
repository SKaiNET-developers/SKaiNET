# scaledDotProductAttention: not recorded by tape, no StableHLO converter

## Summary

`ctx.ops.scaledDotProductAttention()` exists in TensorOps interface,
VoidTensorOps, and DefaultCpuOps — but it is not tape-recorded and has
no StableHLO converter. This blocks multi-head attention in the
SKaiNET → IREE compilation path.

## Impact

Without SDPA, Whisper's multi-head attention must be decomposed into
individual ops (reshape, transpose, matmul, softmax, matmul). The
per-batch K transpose requires raw FloatArray manipulation which
creates zero constants in the VMFB (proven in TapeAttentionPermuteBugTest).

Result: GPU Whisper encoder produces wrong hidden states → decoder
outputs "," instead of real transcription.

## Three fixes needed

### 1. RecordingExecution: record SDPA

**File:** `skainet-compile-core/.../tape/RecordingExecution.kt` line 436

Current (just delegates, no recording):
```kotlin
override fun <T : DType, V> scaledDotProductAttention(...) =
    base.scaledDotProductAttention(query, key, value, mask, scale, causal)
```

Fix (same pattern as conv1d in PR #532):
```kotlin
override fun <T : DType, V> scaledDotProductAttention(
    query, key, value, mask, scale, causal
): Tensor<T, V> {
    val out = base.scaledDotProductAttention(query, key, value, mask, scale, causal)
    val params = mapOf("scale" to scale, "causal" to causal)
    record(ScaledDotProductAttentionOperation(params),
           listOfNotNull(query, key, value, mask), listOf(out))
    return out
}
```

### 2. TensorOperations: add ScaledDotProductAttentionOperation

**File:** `skainet-lang-core/.../tensor/ops/TensorOperations.kt`

```kotlin
class ScaledDotProductAttentionOperation<T : DType, V>(
    parameters: Map<String, Any> = emptyMap()
) : BaseOperation<T, V>("scaledDotProductAttention", "nn", parameters) {
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        // Output shape = query shape: [batch, nHeads, seqLen, headDim]
        return listOf(TensorSpec("sdpa_output", inputs[0].shape, inputs[0].dtype))
    }
}
```

### 3. StableHLO converter: decompose SDPA

**File:** `skainet-compile-hlo/.../converters/NeuralNetOperationsConverter.kt`

Register "scaledDotProductAttention" and decompose into:
```mlir
// scores = Q @ K.T (batched matmul with K transposed)
%scores = stablehlo.dot_general %query, %key,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [3] x [3]
    : (tensor<BxHxSxDxf32>, tensor<BxHxTxDxf32>) -> tensor<BxHxSxTxf32>

// scale
%scaled = stablehlo.multiply %scores, %scale_splat

// optional mask (additive)
%masked = stablehlo.add %scaled, %mask  // if mask != null

// softmax over last dim
%weights = stablehlo softmax ...

// output = weights @ V (batched matmul)
%output = stablehlo.dot_general %weights, %value,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [3] x [2]
```

Note: `contracting_dims = [3] x [3]` for Q@K.T because we contract
headDim of Q (last dim) with headDim of K (also last dim). This is
different from standard matmul where you contract last of A with
second-to-last of B — here K is NOT pre-transposed.

## Test

```kotlin
val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
val q = ctx.fromFloatArray(Shape(1, 6, 4, 64), ...)  // [batch, heads, seq, headDim]
val k = ctx.fromFloatArray(Shape(1, 6, 4, 64), ...)
val v = ctx.fromFloatArray(Shape(1, 6, 4, 64), ...)

val (tape, out) = ctx.record {
    ctx.ops.scaledDotProductAttention(q, k, v)
}

val graph = tape!!.toComputeGraph(synthesizeExternalInputs = true)
val module = StableHloConverterFactory.createExtended().convert(graph, "test_sdpa")

// Should contain dot_general for Q@K.T and weights@V
assertTrue(module.content.contains("stablehlo.dot_general"))
assertFalse(module.content.contains("dense<0.0>"))  // no zero constants
```
