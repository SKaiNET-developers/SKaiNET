# SKaiNET Upstream Issues — Whisper IREE Pipeline

Two issues block the native SKaiNET DSL → StableHLO → IREE compilation path.

## Issue A: Conv1dOperation.inferOutputs echoes input shape

`Conv1dOperation.inferOutputs()` returns `inputs[0].shape` instead of
computing `[batch, outChannels, outLength]`. Same bug in Conv2d/Conv3d.

**File:** `skainet-lang/skainet-lang-core/.../tensor/ops/TensorOperations.kt`
**Fix:** Use `ConvShapeUtils` (already in JAR) from `inferOutputs()`.

## Issue B: toComputeGraph loses edge wiring and op types

`tape.toComputeGraph()` produces nodes where:
- Binary ops (add, matmul, subtract, ...) have wrong input edge count
- Some ops have `operation.type = "trace"` instead of recognized names

157 of 296 Whisper encoder nodes emit "Unsupported ... arity" in MLIR.

**File:** `skainet-compile/skainet-compile-dag/.../tape/extensions.kt` or
`DefaultExecutionTape.toComputeGraph()`

## Test

`Conv1dTapeToHloTest.kt` is a KMP commonTest that:
1. Builds a tape-recording context
2. Runs conv1d → gelu → add through `ctx.ops`
3. Converts tape to ComputeGraph
4. Exports to StableHLO MLIR
5. Asserts: no `tensor<?`, no `Unsupported`, valid `stablehlo.convolution`

Place in: `skainet-compile/skainet-compile-hlo/src/commonTest/kotlin/sk/ainet/compile/hlo/`

Run: `./gradlew :skainet-compile:skainet-compile-hlo:allTests --tests "*Conv1dTapeToHloTest*"`

Currently **fails** on both issues. Will **pass** when both are fixed.
