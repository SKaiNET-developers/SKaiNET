[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.core/skainet-lang-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)
[![REUSE status](https://api.reuse.software/badge/github.com/SKaiNET-developers/SKaiNET)](https://api.reuse.software/badge/github.com/SKaiNET-developers/SKaiNET)
[![OpenSSF Scorecard Score](https://api.scorecard.dev/projects/github.com/SKaiNET-developers/SKaiNET/badge)](https://scorecard.dev/viewer/?uri=github.com/SKaiNET-developers/SKaiNET)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14004/badge)](https://www.bestpractices.dev/projects/14004)


<img src="docs/modules/ROOT/images/SKaiNET-logo.png" alt="SKaiNET logo" width="150">

<a href="https://skainet-developers.github.io/SKaiNET/skainet/reference/architecture.html" title="Open the full architecture reference">
  <img src="docs/modules/ROOT/images/SKaiNET-compiler.png" alt="SKaiNET compiler architecture — click for the full architecture reference" width="640">
</a>

_Click the diagram for the full [architecture reference](https://skainet-developers.github.io/SKaiNET/skainet/reference/architecture.html), or read the short [ARCHITECTURE.md](ARCHITECTURE.md)._

---

## Start in 5 minutes

SKaiNET is a Kotlin Multiplatform AI framework. New here? Choose the path that
matches what you want to try first.

| Goal | Start here | Time |
|---|---|---:|
| Run tensor operations | [Quickstart](#quickstart) (below) | 2–5 min |
| Build and train a neural net | [Hello Neural Net](#hello-neural-net) (below) | 5 min |
| Run a local GGUF model | [SKaiNET Transformers starter](https://github.com/SKaiNET-developers/SKaiNET-transformers#start-in-5-minutes) | 5 min after model setup |
| Export a secure MCU bundle | [Minerva getting started](docs/modules/ROOT/pages/tutorials/minerva-getting-started.adoc) | 10 min without firmware flashing |
| Read low-bit weights (4-bit, ternary) | [Weights at 1.7 bits](docs/modules/ROOT/pages/tutorials/low-bit-weights-getting-started.adoc) | 10 min |

Working in Java? SKaiNET ships first-class Java support — see the
[Java getting-started guide](docs/modules/ROOT/pages/tutorials/java-getting-started.adoc).

> [!NOTE]
> **Looking for LLM inference?** Llama, Qwen, Gemma, Apertus, BERT embeddings and
> GGUF chat models live in
> [**SKaiNET-transformers**](https://github.com/SKaiNET-developers/SKaiNET-transformers) —
> this repository is the engine underneath it (tensors, NN DSL, compiler, CPU/native
> backends, GGUF/SafeTensors IO). Depend on the `sk.ainet.transformers` artifacts,
> pinned together by the
> [transformers BOM](https://central.sonatype.com/artifact/sk.ainet.transformers/skainet-transformers-bom).

Use the version shown in this README as the source of truth for first-run snippets.
If another page shows a different version, please open an issue or PR.

---

## Quickstart

Add the core dependencies (Gradle Kotlin DSL):

```kotlin
dependencies {
    // Recommended: import the umbrella BOM and drop versions on the engine modules.
    implementation(platform("sk.ainet:skainet-bom:0.65.0"))

    implementation("sk.ainet.core:skainet-lang-core")
    implementation("sk.ainet.core:skainet-backend-cpu")
}
```

### Hello Neural Net

```kotlin
val model = nn {
    input(28 * 28)
    dense(out = 128)
    relu()
    dense(out = 10)
}
```

### Core Tensor Ops

```kotlin
val a = tensor(shape(2, 2)) { float(1f, 2f, 3f, 4f) }
val b = tensor(shape(2, 2)) { float(5f, 6f, 7f, 8f) }

val c = a matMul b
val d = c.relu()
```

### GGUF Model Loading

```kotlin
// Recommended: streaming reader — memory-efficient, supports quantized types
val source = JvmRandomAccessSource.open("model.gguf")
StreamingGGUFReader.open(source).use { reader ->
    println("Tensors: ${reader.tensorCount}")

    // Load specific tensor on demand (no whole-file loading)
    val bytes = reader.loadTensor("token_embd.weight")

    // Or get a view over the tensor's bytes: its format, and values decoded on read
    val view = reader.loadTensorView("token_embd.weight")
}
```

> **Full ggml 1.0 type coverage:** every quantization type a ggml writer can put into a GGUF file
> loads — legacy and K-quants kept packed, the IQ/MXFP4/NVFP4/ternary codebook formats decoded
> bit-identical to llama.cpp — and the tokenizer vocabulary is read from the same file.
> See [How SKaiNET handles GGUF](https://skainet-developers.github.io/SKaiNET/skainet/explanation/gguf-handling.html).

> **More examples:** [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook)

---

## Ecosystem

SKaiNET is a modular ecosystem. While this repository contains the core engine, specialized high-level libraries are maintained in standalone repositories:

| Project | Description |
|---|---|
| [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers) | Pre-built transformer architectures and layers |
| [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) | Sample projects and integration demos |

---

## Explore

| Goal | Start here |
|---|---|
| Examples and sample projects | [SKaiNET-examples](https://github.com/SKaiNET-developers/SKaiNET-examples) |
| Interactive notebooks | [SKaiNET-notebook](https://github.com/SKaiNET-developers/SKaiNET-notebook) |
| Eager backends & kernels (what runs where) | [Backends & kernels map](docs/modules/ROOT/pages/explanation/eager-execution.adoc) |
| Design proposals and long-lived API decisions | [SKEEP proposals](docs/modules/skeep/pages/index.adoc) |
| Memory & storage architecture (storage, views, scopes, planning) | [The memory model](docs/modules/ROOT/pages/explanation/memory-model.adoc) · [Packed weight layout](docs/modules/ROOT/pages/explanation/packed-weight-layout.adoc) |

---

## Contributing and Design Proposals

**First time here?** Three links are all you need:

- 🚀 [Getting started as a contributor](https://skainet-developers.github.io/SKaiNET/skainet/contributing/getting-started.html) — the two workflows in one page, and how to claim a task.
- 🟣 [Open good first issues](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3A%22good+first+issue%22) — filter further by what you know: [`skill:android`](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3Askill%3Aandroid), [`skill:numerics` (no Kotlin)](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3Askill%3Anumerics), [`skill:docs`](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3Askill%3Adocs), or by size: [`size:xs`](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3Asize%3Axs).
- 🏷️ [Issue taxonomy](https://skainet-developers.github.io/SKaiNET/skainet/contributing/issue-taxonomy.html) — what every label and `[Lane N · skill]` title prefix means.

Small fixes can go straight through the normal contribution flow described in
[CONTRIBUTING.md](CONTRIBUTING.md) and [GITFLOW.adoc](GITFLOW.adoc).

Non-trivial work goes through one of two processes:

- **DARC** (Document / Assess / Research / Code) for *one feature* — a new
  operator, metric, layer, format reader, or kernel strategy. A feature is one
  parent issue plus skill-labelled sub-issues ("lanes"). See the
  [DARC workflow](https://skainet-developers.github.io/SKaiNET/skainet/contributing/darc-workflow.html).
- **SKEEP** (SKaiNET Evolution and Enhancement Process) for *one architectural
  decision* — public APIs, DSL syntax, tensor semantics, compiler/runtime
  integration, storage behavior, compatibility policy. SKEEP files live under
  `docs/modules/skeep/pages/` with three-digit numbering. See the
  [SKEEP index](https://skainet-developers.github.io/SKaiNET/skainet/skeep/index.html).

---

## Official Benchmarks

SKaiNET ships an official Phoronix-Test-Suite-compatible benchmark
program for the compute engine. See the
[methodology and replay docs](docs/modules/ROOT/pages/contributing/benchmarks.adoc),
the [release manifest](benchmarks/manifests/engine-release.yml), and the
[CI workflow](.github/workflows/engine-benchmarks.yml). Smoke runs fire
on every PR via `ubuntu-latest`; full publishable runs fire on a
self-hosted Linux x86 runner on release.

Quick local replay:

```bash
./gradlew :skainet-backends:benchmarks:jvm-cpu-publish:shadowJar
./scripts/run_engine_smoke.sh
```

---

## Architecture goal

SKaiNET is built around one path: **a model is defined once in the Kotlin DSL,
then either compiled or executed eagerly — without rewriting it.**

1. **Define** the model with the DSL (`nn { }` / `dag { }`).
2. **Capture** it as a *tape* (traced execution) or a *DAG* (explicit graph) — a `ComputeGraph`.
3. **Run** it one of two ways:
   - **Compile** — lower the captured `ComputeGraph` through one of several
     **sibling code-generation backends**, each emitting code for a different target
     from the *same* graph:
     - **StableHLO / MLIR** (`HloGenerator`) → IREE-compilable, for native / edge /
       accelerator targets and the wider MLIR ecosystem.
     - **Arduino / C99** → standalone, statically-allocated C for microcontrollers.
     - **Minerva** → a secure-MCU bundle (weights + firmware skeleton + fingerprinted
       manifest).
   - **Eager** — execute directly on an available backend. On the **JVM this is
     the primary, go-to path.**

StableHLO/MLIR is therefore **one code-generation backend among siblings** — the
IREE/native path next to the C99/Arduino and Minerva MCU paths — not a separate
pipeline.

```mermaid
flowchart LR
    DSL["Model — Kotlin DSL"] --> Graph["Tape / DAG (ComputeGraph)"]
    Graph --> Eager["Eager backend (JVM, …)"]
    Graph -->|code generation| HLO["StableHLO / MLIR"]
    Graph -->|code generation| C99["Arduino / C99"]
    Graph -->|code generation| Minerva["Minerva"]
    HLO --> Native["IREE → native / edge / accelerator"]
    C99 --> MCU["Microcontroller"]
    Minerva --> SecMCU["Secure-MCU bundle"]
```

The same DSL model feeds every path: eager execution for development and JVM
deployment, and the code-generation backends — StableHLO/MLIR (→ IREE), Arduino/C99,
and Minerva — as **sibling alternatives** for native, edge, and secure-MCU targets.

---

## Important Addition: Minerva Secure MCU Export

SKaiNET now includes a Minerva export backend for secure MCU deployment. It is a sibling to StableHLO and Arduino/C99 export: it starts from a supported `ComputeGraph`, lowers static MLPs to a Minerva compiler input, invokes libminerva when configured, and packages generated weights, host fixtures, firmware skeletons, and a fingerprinted `manifest.json`.

Start here:

- [Minerva getting started](docs/modules/ROOT/pages/tutorials/minerva-getting-started.adoc) — run the maintained tiny MLP dry sample, then the real libminerva runtime profile.
- [Minerva export how-to](docs/modules/ROOT/pages/how-to/minerva-export.adoc) — configure compiler paths, keys, calibration, CMake/CTest host verification, and troubleshooting.
- [How Minerva secure MCU export fits](docs/modules/ROOT/pages/explanation/minerva-secure-mcu-export.adoc) — understand why Minerva is not an Arduino replacement and when to choose StableHLO instead.

Runnable examples:

```bash
./gradlew :skainet-compile:skainet-compile-minerva:runMinervaSecureMcuExamples
./gradlew :skainet-compile:skainet-compile-minerva:runMinervaSecureMcuExamples \
  -Pminerva.example=sensor-classifier
```

---

## Features

### Kotlin Multiplatform

- Targets: JVM, macOS (Native), JS, WASM (Browser + WasmWasi)
- Single codebase shared across all platforms via Kotlin Multiplatform

### Optimized Execution

- **ComputeGraphExecutor**: Optimized engine with fusion passes and trace-to-DAG bridging.
- **SDPA & Gather**: High-performance Scaled Dot-Product Attention and indexing operations.
- **TurboQuant**: Runtime KV-cache compression (~8x at 4-bit) for long-context LLM inference. Presets: `safe-lowbit`, `balanced`, `experimental-max`. See `TurboQuantUsage` for integration guide.

### Neural Network DSL

- **Sequential**: `nn { input(); dense(); relu(); dense() }`
- **DAG / Graph**: arbitrary wiring with `dag { }` for ResNet-style skip connections and multi-scale detector heads
- Layers: Dense, Conv1d/2d/3d, MaxPool, AvgPool, BatchNorm, Dropout, LeakyReLU, ELU
- KAN (Kolmogorov–Arnold Networks) layer (experimental)
- Autograd engine with reverse-mode gradients, SGD and Adam/AdamW optimizers

### Data and I/O

- Built-in loaders: MNIST, Fashion-MNIST, CIFAR-10, Iris
- URI-backed data sources: `file://`, `https://`, `hf+https://`, and `hf://...`
- Dataset operations: deterministic shuffle/split, stratified split, filter/map/transform views, batch flows, and epoch flows
- Raw dataset parsers: CSV, TSV, JSON arrays/objects, JSON Lines (`.jsonl`, `.ndjson`)
- Type-safe transform DSLs: image/tensor transforms plus suspendable raw data pipelines
- Formats: GGUF, ONNX, SafeTensors, JSON, Image (JPEG, PNG)
- GGUF: streaming reads with metadata-only inspection, full ggml 1.0 quantization type coverage (bit-identical to llama.cpp), in-file tokenizer vocabulary

```kotlin
val raw = JvmDataSourceResolver().rawDataset {
    from("hf://datasets/org/repo@main/train.jsonl")
    format(DataFormat.JSON_LINES)
    cachePolicy(CachePolicy.Use)
}

val withoutLabel = dataPipeline<RawDataset>()
    .stage(
        dataTransformer(
            name = "drop-label",
            outputSchema = { schema -> DataSchema(schema.columns - "label") }
        ) { dataset ->
            val columns = dataset.schema.columns - "label"
            dataset.copy(
                schema = DataSchema(columns),
                rows = dataset.rows.map { row ->
                    RawDataRow(row.values.filterKeys { key -> key in columns })
                }
            )
        }
    )
    .execute(raw)
```

- Start with the [data sources getting started guide](docs/modules/ROOT/pages/tutorials/data-sources-getting-started.adoc)

### Edge AI: Arduino / C99 Export

- Export trained models to standalone, optimized C99 with static memory allocation
- Ready-to-use Arduino library output

### Edge AI: Minerva Secure MCU Export

- Export supported static MLP graphs to Minerva project bundles for secure MCU inference
- Emits compiler NPZ input, libminerva weights, a fingerprinted manifest, host harness, firmware example, and host verification results
- Start with the [Minerva getting started guide](docs/modules/ROOT/pages/tutorials/minerva-getting-started.adoc)

### Compiler: MLIR / StableHLO

- Lower Kotlin DSL to MLIR StableHLO dialect
- Optimization passes: constant folding, operation fusion, dead code elimination
- Valid IREE-compilable output with streaming API and public `HloGenerator`

### Choosing an Export Path

- Use **StableHLO** when you want portable MLIR/IREE-compatible graphs for native, accelerator, or ecosystem compiler flows.
- Use **Arduino / C99 export / Minerva export** when you want standalone generated C with static memory allocation or external secure runtime.

---

## What's New in 0.65.0

One storage model: weights stay in their packed form from the file to the kernel, and the old storage layer is
removed (SKEEP-003). This release has breaking changes; the changelog gives the migration for each removed type.

- **One view for every weight format** — `StreamingGGUFReader` and `StreamingSafeTensorsReader` return a
  `TensorView` (`loadTensorView`, `loadTensorViewMapped`) over a tensor's bytes, without decoding or copying them.
  `view.format` says what the bytes are (`Float32/Q4_0`, `Float32/TQ1_0`, `Float16/Dense(2B)`), `get` and
  `toFloatArray` decode on read, and a mapped view works for a tensor above 2 GB.
- **Kernels are chosen from the format** — `matmul` with a packed weight looks its kernel up by `KernelKey` from the
  two views, on every backend. The ops no longer check the data's class: 162 downcasts became 8, and a build task
  keeps the count from rising.
- **Packed weights are FP32** — the packed block types are `TensorData<FP32, Float>`; `get` returns the decoded
  value and `getCode` the stored code. `view.requantizeTo(encoding)` is the lossy conversion, `materialize()` the
  lossless one.
- **Your own encoding** — `TensorEncoding.Opaque`, a `BlockDecoder` and a `ViewKernel` add a weight format and its
  matmul kernel from outside the library.
- **New docs** — the tutorial "Weights at 1.7 bits: one view for every format", the how-to "Define your own
  low-bit encoding", and "What the model replaced" in the memory model page. All samples are compiled and run in CI.
- **Removed** — `TensorStorage`, `BufferHandle`, `LogicalDType`, `SlicedTensorView` and the old `TensorView`
  marker, `loadTensorStorage*`, the two `MemorySegment` packed weight classes. `slice()` and `sliceView { }`
  return a `Tensor`.
- **Deprecated** — `QuantizedMatmul` and `TernaryMatmul`; use `ops.matmul(input, weight)`.
- **Also** — optional axis labels on `Shape`, a debug-mode warning for element reads of a packed view, activation
  ids for eager execution in debug mode, and a fix for `ops.matmul` with a ternary right-hand weight.

See [CHANGELOG.md](CHANGELOG.md) for full release notes, including every prior release.

---

## Roadmap

- **Q1 2026**: Comprehensive documentation ✅
- **Q2 2026**: TurboQuant KV-cache compression ✅ (shipped in 0.18.0); Qwen/LLaMA tokenizers ✅ (shipped in 0.20.0)
- **Q3 2026**: Missing ML features: metrics, optimizers, and training utilities. 
- **Q4 2026**: On-Device AI, small LLMs improvements

---


## Contributing & Community

We love contributions! Whether it's a new operator, documentation, or a bug fix:

1. Read [Getting started as a contributor](https://skainet-developers.github.io/SKaiNET/skainet/contributing/getting-started.html) (five minutes), then the [Contribution Guide](CONTRIBUTING.md) when you need the procedure.
2. Pick an [open good first issue](https://github.com/SKaiNET-developers/SKaiNET/issues?q=is%3Aopen+label%3A%22good+first+issue%22) — every one names the file to copy the pattern from and the exact Gradle task to run. Comment on it to claim it.
3. Open a discussion or issue on [GitHub](https://github.com/SKaiNET-developers/SKaiNET/issues); the issue chooser has templates for DARC features, lane tasks and SKEEP proposals.

Browse the full codebase documentation on [DeepWiki](https://deepwiki.com/SKaiNET-developers/SKaiNET).

### Contributors (0.65.0)

- **Michal Harakal** ([@michalharakal](https://github.com/michalharakal)) — the finish of SKEEP-003: kernel
  dispatch by format on every backend, views from the loaders, the removal of the old storage layer, the low-bit
  tutorial and how-to, and the release

---

## Citing

If you use SKaiNET in your research, please cite it. Citation metadata lives in
[`CITATION.cff`](CITATION.cff) (GitHub's *Cite this repository* button exports BibTeX and APA
from it). Ready-to-paste BibTeX:

```bibtex
@misc{skainet,
  author       = {Harakal, Michal and {SKaiNET-developers contributors}},
  title        = {{SKaiNET}: A {Kotlin} Multiplatform {AI} Framework},
  year         = {2026},
  note         = {Version 0.65.0. MIT License},
  howpublished = {\url{https://github.com/SKaiNET-developers/SKaiNET}}
}
```

Please include the version you used (each release is tagged).

## License

MIT — see [LICENCE](LICENCE).
