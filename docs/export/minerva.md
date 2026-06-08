# Minerva Secure MCU Export

Minerva export packages a supported SKaiNET compute graph for secure MCU inference through libminerva. The maintained docs-site version is [`docs/modules/ROOT/pages/how-to/minerva-export.adoc`](../modules/ROOT/pages/how-to/minerva-export.adoc); this Markdown entrypoint keeps the repository path requested by the planning issue and is friendly to GitHub browsing.

## Setup

Inside this repository, use `project(":skainet-compile:skainet-compile-minerva")`. Published applications should import the SKaiNET BOM and add `sk.ainet.core:skainet-compile-minerva`.

Configure libminerva through `MinervaExportOptions` or the JVM sample environment:

```bash
export MINERVA_COMPILER_SCRIPT=/opt/libminerva/tools/compile_model.py
export MINERVA_RUNTIME_ROOT=/opt/libminerva
export MINERVA_CALIBRATION_NPZ=/secure/project/calibration.npz
export MINERVA_KEY_FILE=/secure/project/device.key
export MINERVA_RUN_CMAKE=true
export MINERVA_RUN_CTEST=true
```

Do not commit real device keys. `include/secrets.example.h` contains placeholders only.

## Compatibility

| Area | Phase-one support |
|---|---|
| Host platform | JVM export path |
| Target | `MinervaTarget.ATMEGA328P` |
| Quantization | `MinervaQuantization.Q8` |
| Graphs | Static, single-path, sequential MLPs |
| Shapes | Fully known rank-2 tensor shapes |
| Pattern | `Input -> MatMul -> Add? -> activation?`, repeated in sequence |
| Activations | `Relu`, `Sigmoid`, `Tanh` after a dense layer |
| Out of scope | CNNs, attention, recurrent models, dynamic shapes, branching graphs, transformers, arbitrary ONNX operators |

## Export API

```kotlin
val options = MinervaExportOptions(
    outputDir = "build/minerva",
    projectName = "TinySecureMlp",
    compilerScript = "/opt/libminerva/tools/compile_model.py",
    runtimeRoot = "/opt/libminerva",
    calibrationNpz = "/secure/project/calibration.npz",
    keyFile = "/secure/project/device.key"
)

val result = MinervaExportFacade().exportGraph(graph, options)
val bundle = result.requireSuccess()
println(bundle.outputDir)
```

If the compiler script is missing, export still runs compatibility validation, lowering, and NPZ generation before returning a typed compiler prerequisite failure.

## Generated Layout

```text
build/minerva/TinySecureMlp/
  manifest.json
  generated/
    model.npz
    weights.c
  include/
    weights.h
    secrets.example.h
  host/
    CMakeLists.txt
    main.c
  firmware/
    main.c
```

## Host Verification and CI

Host verification checks package structure, generated weight files, `model.npz` integrity, placeholder secret hygiene, and SKaiNET reference output generation. Add these metadata keys to opt into CMake, CTest, and parity comparison with a host output file:

```kotlin
metadata = mapOf(
    MinervaHostVerificationMetadata.RUN_CMAKE_BUILD to "true",
    MinervaHostVerificationMetadata.RUN_CTEST to "true",
    MinervaHostVerificationMetadata.HOST_OUTPUT_PATH to "host-output.txt"
)
```

CI recipe:

```bash
./gradlew :skainet-compile:skainet-compile-minerva:jvmTest
./gradlew :skainet-compile:skainet-compile-minerva:minervaHostVerification \
  -Pminerva.hostVerification.enabled=true \
  -Pminerva.runtimeRoot="$MINERVA_RUNTIME_ROOT" \
  -Pminerva.compilerScript="$MINERVA_COMPILER_SCRIPT" \
  -Pminerva.calibrationNpz="$MINERVA_CALIBRATION_NPZ" \
  -Pminerva.keyFile="$MINERVA_KEY_FILE"
```

`minervaHostVerification` is skipped by default. When enabled, it runs `jvmTest` and `runMinervaTinyMlpSample` with CMake and CTest host verification enabled unless `-Pminerva.hostVerification.runCmakeBuild=false` or `-Pminerva.hostVerification.runCTest=false` is set.

## ONNX to Minerva

Use the existing ONNX loader to inspect a model and reject unsupported operators before constructing a compatible SKaiNET `ComputeGraph`:

```kotlin
val loaded = OnnxLoader.fromModelSource {
    File(path).inputStream().asSource()
}.load()
val graph = loaded.proto.graph ?: error("ONNX model has no graph")
val ops = graph.node.map { it.opType }.toSet()
require(ops.all { it in setOf("MatMul", "Gemm", "Add", "Relu", "Sigmoid", "Tanh") })
```

The first phase does not include a general ONNX-to-Minerva importer.

## Firmware Integration

The generated firmware example intentionally contains placeholders. Confirm the libminerva inference entry point and output-authentication API names against the pinned libminerva version used by your product build before flashing firmware.

## Maintained JVM Sample

`sk.ainet.compile.minerva.examples.MinervaTinyMlpExportSample` builds a tiny two-layer MLP, reads Minerva paths from environment variables or Gradle properties, invokes the export facade, and prints bundle and verification status.

```bash
./gradlew :skainet-compile:skainet-compile-minerva:runMinervaTinyMlpSample
```

Without `MINERVA_COMPILER_SCRIPT`, the task runs a dry validation through compatibility, lowering, and in-memory NPZ generation. Add `-Pminerva.compilerScript`, `-Pminerva.runtimeRoot`, `-Pminerva.calibrationNpz`, and `-Pminerva.keyFile` to run the real compiler path. `MinervaTinyMlpExportSampleTest` validates the sample graph and NPZ generation without real device keys.

## Export Path Choice

- Use StableHLO for portable MLIR/IREE-compatible compiler flows.
- Use Arduino / C99 export for standalone generated C with static memory allocation.
- Use Minerva export for secure MCU bundles compiled by libminerva and checked by host verification.
