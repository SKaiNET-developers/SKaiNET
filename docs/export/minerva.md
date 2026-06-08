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
export MINERVA_HOST_ADAPTER_SOURCE=/secure/project/minerva_runtime_adapter.c
export MINERVA_HOST_INCLUDE_DIRS=/opt/libminerva/include
export MINERVA_HOST_LIBRARY_DIRS=/opt/libminerva/lib
export MINERVA_HOST_LIBRARIES=minerva
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
| Out of scope | CNNs, attention, recurrent models, dynamic shapes, branching graphs, transformers, and arbitrary imported operator sets |

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
    runtime_adapter.example.c
    reference-input.txt
    reference-output.txt
    observed-output.txt   # optional output from a real host run
  firmware/
    main.c
```

## Manifest Provenance

`manifest.json` records the export target, quantization, libminerva root, compiler command summary, NPZ schema version, layer count, reference fixture paths, generated files, and a `generatedFileSha256` map. The hashes cover generated artifacts such as `generated/model.npz`, `generated/weights.c`, `include/weights.h`, host fixtures, host harness sources, and firmware examples.

Use the manifest as the handoff record between the source model, Kotlin export, libminerva compilation, and host verification. It redacts compiler key-file arguments and does not copy real device key material; keep real keys outside the generated bundle.

## Host Verification and CI

Host verification checks package structure, generated weight files, `model.npz` integrity, placeholder secret hygiene, and SKaiNET reference fixture generation. The packager writes deterministic `host/reference-input.txt` and `host/reference-output.txt` files and records them in `manifest.json`. A real host run can write comma- or whitespace-separated float outputs to `host/observed-output.txt` for zero-config parity comparison.

The generated host harness has a stable adapter ABI:

```c
int minerva_run_inference(const float *input, int input_count, float *output, int output_count);
```

`host/runtime_adapter.example.c` implements that ABI against the libminerva public C API (`mnv_init`, `mnv_seed_prng`, `mnv_run`, and `mnv_verify_output`). The adapter converts SKaiNET's normalized float fixtures to libminerva Q8 activation buffers and converts Q8 outputs back to floats for parity comparison.

Copy the adapter outside the generated bundle when product-specific scaling or entropy seeding needs local edits, then point CMake at the copied source. This keeps the generated host harness stable while leaving runtime policy in one reviewable adapter file.

Add these metadata keys to opt into CMake, CTest, and parity comparison with a custom host output file:

```kotlin
metadata = mapOf(
    MinervaHostVerificationMetadata.RUN_CMAKE_BUILD to "true",
    MinervaHostVerificationMetadata.RUN_CTEST to "true",
    MinervaHostVerificationMetadata.HOST_OUTPUT_PATH to "host-output.txt",
    MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE to "/secure/project/minerva_runtime_adapter.c",
    MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS to "/opt/libminerva/include",
    MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS to "/opt/libminerva/lib",
    MinervaHostVerificationMetadata.HOST_LIBRARIES to "minerva"
)
```

`HOST_OUTPUT_PATH` is optional when the host run writes `host/observed-output.txt`.
The include, library directory, and library values are passed to CMake as semicolon-separated lists, matching CMake list syntax.

CI recipe:

```bash
./gradlew :skainet-compile:skainet-compile-minerva:jvmTest
./gradlew :skainet-compile:skainet-compile-minerva:minervaHostVerification \
  -Pminerva.hostVerification.enabled=true \
  -Pminerva.runtimeRoot="$MINERVA_RUNTIME_ROOT" \
  -Pminerva.compilerScript="$MINERVA_COMPILER_SCRIPT" \
  -Pminerva.calibrationNpz="$MINERVA_CALIBRATION_NPZ" \
  -Pminerva.keyFile="$MINERVA_KEY_FILE" \
  -Pminerva.hostVerification.hostAdapterSource="$MINERVA_HOST_ADAPTER_SOURCE" \
  -Pminerva.hostVerification.hostIncludeDirs="$MINERVA_HOST_INCLUDE_DIRS" \
  -Pminerva.hostVerification.hostLibraryDirs="$MINERVA_HOST_LIBRARY_DIRS" \
  -Pminerva.hostVerification.hostLibraries="$MINERVA_HOST_LIBRARIES"
```

`minervaHostVerification` is skipped by default. When enabled, it runs `jvmTest` and `runMinervaTinyMlpSample` with CMake and CTest host verification enabled unless `-Pminerva.hostVerification.runCmakeBuild=false` or `-Pminerva.hostVerification.runCTest=false` is set.

## Model Sources

Minerva export starts from a supported SKaiNET `ComputeGraph`. That graph can come from the Kotlin DSL, a traced forward pass, a hand-built graph, an imported model, or any other path that preserves the phase-one static MLP contract.

For ONNX inputs, use the existing ONNX loader as an inspection step before constructing a compatible SKaiNET graph:

```kotlin
val loaded = OnnxLoader.fromModelSource {
    File(path).inputStream().asSource()
}.load()
val graph = loaded.proto.graph ?: error("ONNX model has no graph")
val ops = graph.node.map { it.opType }.toSet()
require(ops.all { it in setOf("MatMul", "Gemm", "Add", "Relu", "Sigmoid", "Tanh") })
```

The first phase does not include a general ONNX-to-Minerva importer. Once any source graph is exported, keep `manifest.json` with the source-model artifact or training metadata so reviewers can compare provenance, generated `model.npz`, and host verification fixtures.

## Firmware Integration

The generated firmware example intentionally contains integration placeholders. Use the host adapter as the reference for the pinned libminerva public API, then move product-specific entropy seeding, input scaling, and secret provisioning into private firmware code before flashing.

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
