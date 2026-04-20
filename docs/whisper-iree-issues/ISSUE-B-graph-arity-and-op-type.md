# toComputeGraph() loses edge wiring and produces wrong op types

## Problem

After `tape.toComputeGraph(synthesizeExternalInputs = true)`, many graph nodes have:

1. **Wrong input edge count** — binary ops (add, matmul, subtract) don't get 2 input
   edges wired; unary ops (gelu, softmax, reshape) don't get 1 input edge wired.
   The StableHLO converter checks arity and emits "Unsupported X arity" comments.

2. **Wrong operation type** — some ops have `operation.type = "trace"` instead of a
   recognized category. The converter uses `operation.name` for dispatch but some
   converters also check `type`, and "trace" doesn't match any registered converter.

## Scope

Whisper encoder tape produces 296 graph nodes. After StableHLO conversion:
- 166 nodes emit valid `stablehlo.*` ops
- 157 nodes emit `// Unsupported ...` comments (some nodes emit both)

Breakdown of unsupported:

```
 32  add             — wrong arity (expected 2 inputs)
 24  matmul          — wrong arity (expected 2 inputs)
 16  unsqueeze       — wrong arity (expected 1 input)
 12  reshape         — wrong arity (expected 1 input)
  9  subtract        — wrong arity (expected 2 inputs)
  9  sqrt            — type "trace" not recognized
  9  addScalar       — type "trace" not recognized
  9  multiply        — wrong arity (expected 2 inputs)
  9  divide          — wrong arity (expected 2 inputs)
  7  variance        — wrong arity (expected 1 input)
  7  mean            — wrong arity (expected 1 input)
  4  softmax         — wrong arity (expected 1 input)
  4  mulScalar       — type "trace" not recognized
  4  gelu            — wrong arity (expected 1 input)
  2  mean            — type "trace" not recognized
```

## Root cause hypothesis

`DefaultExecutionTape.toComputeGraph()` builds graph edges by matching tensor ref
IDs between operation outputs and subsequent operation inputs. If:

- The ref ID scheme changed between recording and graph construction, edges don't
  connect and binary ops appear to have 0 or 1 inputs.
- Weight tensors created before `startRecording()` may not have their ref IDs in
  the tape's scope, so edges from weights to consumers are missing.

The `type = "trace"` issue: `KspTensorOps` (the auto-generated tracing wrapper)
may record operations with a generic "trace" type string for ops that don't have
an explicit `Operation` subclass (sqrt, addScalar, mulScalar, etc.).

## Impact

The generated MLIR is structurally incomplete — most ops are comments instead of
valid StableHLO operations. `iree-compile` cannot process it.

## Suggested investigation

1. In `DefaultExecutionTape.toComputeGraph()`, check how `GraphEdge` source/target
   are resolved from tape trace inputs/outputs. Are ref IDs stable?
2. For the `type = "trace"` ops: check what `KspTensorOps` records as the operation
   type for `sqrt`, `addScalar`, `mulScalar`. The HLO converter's operation
   registry should recognize these names regardless of type.
3. The `synthesizeExternalInputs = true` flag should create input/weight nodes for
   external tensors — verify these get edges to their consumers.

## Test

See `Conv1dTapeToHloTest.kt` — asserts that `Unsupported` does not appear in
output MLIR for a simple conv1d → gelu → add pipeline.
