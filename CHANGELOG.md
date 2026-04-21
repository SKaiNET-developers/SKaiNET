# Changelog

## [Unreleased]

## [0.19.1] - 2026-04-21

### Fixed

- **Broken POM for `skainet-backend-cpu`**: The 0.19.0 POM for `sk.ainet.core:skainet-backend-cpu-*` declared a runtime dependency on `sk.ainet:skainet-backend-api-jvm:unspecified` — wrong group coordinate and no valid version, because `skainet-backend-api` was not configured to publish and the root `allprojects { group = "sk.ainet" }` disagreed with the `GROUP=sk.ainet.core` used by vanniktech's maven publish plugin. Consumers pulling 0.19.0 hit unresolved-dependency errors. Fixed by:
  - Applying `vanniktech.mavenPublish` and setting `POM_ARTIFACT_ID=skainet-backend-api` on `skainet-backend-api` so it is actually published alongside the BOM entry that already referenced it.
  - Aligning `allprojects { group = "sk.ainet.core" }` with the `GROUP` property and pinning `version` from `VERSION_NAME` so `project(...)` coordinates in generated POMs are consistent.
- **CI guard**: New `verify-published-poms` job publishes to the local Maven repository and fails the build if any generated `.pom` contains `<version>unspecified</version>` or references a project-local group outside `sk.ainet.core`, preventing a regression of this class of coordinate bug.

## [0.19.0] - 2026-04-20

### Added

#### Tokenizers
- **Qwen / GPT-2 Byte-Level BPE Tokenizer**: `QwenByteLevelBpeTokenizer` implements the full GPT-2-style pipeline — byte-to-unicode mapping, GPT-2 pretokenization regex, merge-rank BPE, and atomic special-token splitting. Builds from either GGUF metadata (`fromGgufFields`) or a HuggingFace `tokenizer.json` (`fromTokenizerJson`). Verified against Qwen2.5-0.5B reference token IDs from HuggingFace `transformers`. (#463)
- **LLaMA / SentencePiece Tokenizer**: `SentencePieceTokenizer` implements the llama.cpp SPM pipeline — whitespace escape (`▁`), code-point symbol split, **score-priority** BPE (the SPM rule, opposite of the merge-rank rule used for GPT-2 BPE), and `<0xNN>` byte fallback for unknown characters. Builds from GGUF (`tokenizer.ggml.model == "llama"`) and HuggingFace `tokenizer.json` (`model.type == "Unigram"`). Verified against TinyLlama-1.1B reference token IDs from HuggingFace `transformers`. (#464)
- **`TokenizerFactory` with Per-Architecture Dispatch**: Tokenizer selection is now **per-architecture, not per file format**. `TokenizerFactory.fromGguf(fields)` and `.fromTokenizerJson(json)` inspect `tokenizer.ggml.model` / `model.type` and dispatch to the right implementation — Qwen/GPT-2 → byte-level BPE, LLaMA/Gemma/TinyLlama → SentencePiece — regardless of whether weights come from GGUF or SafeTensors. (#463)
- **`Tokenizer` Interface**: Common surface implemented by `TekkenTokenizer`, `QwenByteLevelBpeTokenizer`, and `SentencePieceTokenizer` (`encode`, `decode`, `vocabSize`, `bosTokenId`, `eosTokenId`).
- **GGUF Tokenizer Metadata**: `GgufModelMetadata` now exposes `tokenizerModel`, `tokenizerTokens`, `tokenizerMerges`, `tokenizerTokenTypes`, `bosTokenId`, and `eosTokenId` so callers can build a tokenizer without re-parsing the raw field map.

#### StableHLO → IREE compilation
- **Whisper Encoder E2E**: Whisper encoder now compiles end-to-end via SKaiNET → StableHLO → IREE.
- **Real StableHLO Lowerings**: `softmax`, `layerNorm`, and `rmsnorm` now lower to real StableHLO ops (reductions, `broadcast_in_dim`, standard ops) instead of `custom_call` stubs. (#467, #479, #480)
- **New Op Converters**: `gather` / `embedding`, and `concat` / `slice` / `cast` StableHLO converters. (#483, #489)
- **Activation Alias**: `silu` / `SiLU` registered as an alias for `swish` in `ActivationOperationsConverter`. (#484)
- **`ConstantMaterializationPolicy`**: Seam for externalizing large weight tensors out of the StableHLO module (enables `.irpa` externalization). (#524)
- **Splat Constant Folding**: Uniform-value tensor constants collapsed to `dense<v>` splat instead of fully materialized arrays. (#522)
- **SSA Value Type Tracking**: Tracks SSA value types so `reshape` emits the operand's declared type, producing valid MLIR. (#521)
- **Tensor Encoding in Output**: `tensor_encoding` comments in StableHLO output and a top-level `skainet.tensor_encodings` module attribute. (#473, #477)

#### IREE `.irpa` weight files
- **`skainet-io-iree-params` Module**: New module with `IrpaWriter` for writing IREE Parameter Archive (`.irpa`) files. Accepts `FileBacked` handles via mmap on JVM / Android for zero-copy weight export. (#523, #525, #528, #529)

#### Backend API
- **`skainet-backend-api` Module**: New module cleanly separating backend contracts; CPU backend now depends on it. (#468)
- **`TensorEncoding` Metadata**: Accessor for `TensorSpec.metadata` and propagation through `TraceToGraphBuilder.finalize`, keeping quantization encoding visible end-to-end. (#469)

#### Java API (0.19.0 surface polish)
- Annotated `StableHloConverterFactory` and `TokenizerFactory` for idiomatic Java call sites. (#400)
- Renamed `TensorSpecEncoding.kt` class for Java callers. (#400)
- Added `skainet-backend-api` to the BOM. (#400)
- New `ReleaseApiJavaTest` covering the 0.19.0 Java surface. (#400)

#### Docs (Antora migration)
- **Antora + Diátaxis**: Migrated docs to Antora with Divio / Diátaxis layout (tutorials, how-tos, reference, explanation). (#494)
- **`skainet-docs-ui` v1.1.1**: Adopted the new theme with Diátaxis card-grid landing page. (#501)
- **Operator Coverage Matrix**: Emit cross-backend Operator Coverage Matrix generated from `TensorOps` surface scan. (#494, #511)
- **Ops Docs**: KDoc `@param` extraction, real version stamps, LaTeX rendering, fixed partials, and dropped void backend. (#511, #513)
- **Dokka API Bundle**: Wired into the Antora site build. (#494)
- **Local Mermaid**: Drop kroki, render Mermaid locally via `mmdc`. (#496)

#### Platform targets
- **`androidNativeArm32`**: Added across core modules. (#503)

### Fixed
- **Byte-Level BPE Broken for Qwen/GPT-2 Models**: Previously there was no GPT-2-style byte-level BPE tokenizer in the repo, and `GgufModelMetadata` ignored `tokenizer.ggml.merges` entirely — so any Qwen / GPT-2 / Mistral-Nemo model encoded text into garbage tokens (byte-level chars instead of merged vocab IDs), blocking chat mode and tool calling. The new `QwenByteLevelBpeTokenizer` + `TokenizerFactory` dispatch fix the issue for both GGUF and SafeTensors sources. (#463)
- **No SentencePiece Path for LLaMA-Family GGUF Models**: `TokenizerFactory` previously threw `UnsupportedTokenizerException` for `tokenizer.ggml.model == "llama"`, leaving LLaMA / TinyLlama / Gemma / Mistral-v0.1 GGUFs untokenizable. The new `SentencePieceTokenizer` closes that gap. (#464)
- **GGUF UInt Fields Silently Dropped**: GGUF UINT32 fields (e.g. `tokenizer.ggml.bos_token_id`) arrive from `StreamingGGUFReader` as `kotlin.UInt`, which is a value class — *not* a subclass of `kotlin.Number` — so a plain `as? Number` cast was returning null. The new `toIntFlexible` helper handles every signed and unsigned numeric type GGUF can produce, restoring the BOS/EOS/UNK ids on the tokenizer builders.
- **Graph Conv Output Shape Inference**: `conv1d` / `conv2d` / `conv3d` operations in graph inference previously produced placeholder output shapes, breaking downstream shape-dependent passes. Graph ops now compute real output shapes. (#536, #537)
- **Conv1d/Conv3d Not Recorded**: `conv1d` and `conv3d` were not routed through the recording decorator, so they disappeared from traced computation graphs. (#532, #533)
- **Static Conv1d HLO Shape Crash**: Conv1d StableHLO lowering crashed when trace attributes were missing; now falls back to `TensorRef` shape / dtype. (#530, #531)
- **Flatten Hardcoded to MNIST Shape**: `NetworkBuilder.flatten()` returned a hardcoded `lastDimension = 1568` (the MNIST CNN value); any other architecture — e.g. a 64-channel CNN over 32×32 inputs — crashed with `ArrayIndexOutOfBoundsException` in the following `dense()` layer. The DSL now tracks per-sample shape through a new `input(IntArray)` overload, `conv1d` / `conv2d` / `conv3d`, `maxPool2d`, `avgPool2d`, and `upsample2d`, reusing the `ConvShapeUtils` arithmetic introduced in #537; `flatten()` reads the tracked shape and honors `startDim` / `endDim`, and `Conv*` layers can auto-infer `inChannels` from the declared input. (#535, #538)
- **StableHLO `transpose` / `dot_general` MLIR Emission**: Fixed malformed MLIR produced by `stablehlo.transpose` and `stablehlo.dot_general` that blocked IREE compilation. (#520)
- **WasmJS / JS / Native Compile**: Replaced JVM-only `putIfAbsent` with a common-stdlib idiom. (#485)
- **Antora Container**: `HOME=/tmp` so Chromium crashpad can launch during Mermaid rendering in CI. (#534)
- **`bundleDokkaIntoSite` CI Permission Failure**: Fixed docs pipeline permission error. (#496)
- **Pandoc Artifacts in Docs**: Stripped pandoc anchors and demoted heading levels in migrated pages. (#496)

### Changed
- **`compile-hlo` Dependencies**: Dropped vestigial `skainet-backend-cpu` dependency from `compile-hlo` jvmMain. (#472)
- **Moved-LLM Docs**: Replaced relocated LLM pages with redirect stubs pointing at the standalone repo. (#499)
- **Maven Group / Version Refs**: Bumped stale version references and fixed Maven group coordinates. (#499)

### Removed
- Stale `TURBOQUANT_ISSUES.md` tracker at the repo root. (#490)

### Dependencies
- agp: 9.1.0 → 9.1.1.
- com.networknt:json-schema-validator: 3.0.1 → 3.0.2.
- org.jetbrains.kotlinx:kotlinx-serialization-json: bumped to 1.11.0.
- actions/checkout: 4 → 6.
- actions/upload-pages-artifact: 3 → 5.
- actions/cache: 4 → 5.
- actions/setup-java: 4 → 5.
- actions/deploy-pages: 4 → 5.
- actions/github-script: 8 → 9.
- docker/build-push-action: 5 → 7.
- docker/setup-buildx-action: 3 → 4.

## [0.18.0] - 2026-04-08

### Added
- **TurboQuant KV-Cache Compression**: Runtime KV-cache compression for LLM inference using rotation-based quantization (Google Research TurboQuant paper). Supports PolarOnly and PolarPlusQjl variants with 2/3/4/8-bit encoding.
  - `TurboQuantCodec`: End-to-end encode/decode pipeline (random rotation, scalar quantization, QJL residual, bit-packing).
  - `TurboQuantKvCacheStore`: Compressed KV cache with per-head TurboQuant blocks and asymmetric K/V policies.
  - `TurboQuantPresets`: Named presets — `safe-lowbit` (Q8_0-K + TQ4-V), `balanced` (TQ4/TQ4), `experimental-max` (TQ3/TQ3).
  - `KvCacheStore.turboQuant("balanced", ...)`: One-line factory for skainet-transformers integration.
  - `CompressedKvAttention`: SDPA bridge with FULL_TILE and RAW_STORAGE dequant strategies.
  - `@KvCache` and `@KvCacheBypass` DSL annotations for declarative KV cache configuration.
  - `KvCacheAnnotationResolver`: Resolve annotations to cache instances.
  - `TurboQuantUsage`: Documented integration guide with compilable examples.
- **Memory Architecture Hardening**: First-class storage and placement abstractions for zero-copy, quantization-preserving tensor management.
  - `TensorStorage`: Runtime descriptor replacing ad-hoc array passing (logical type, physical encoding, buffer ownership, placement).
  - `TensorEncoding`: Sealed hierarchy — `Dense`, `Q4_K`, `Q8_0`, `TernaryPacked`, `TurboQuantPolar`, `TurboQuantPolarQjl`, `Opaque`.
  - `BufferHandle`: Five ownership modes — `Owned`, `Borrowed`, `Aliased`, `FileBacked`, `DeviceResident`.
  - `Placement`: Device/memory-domain intent with fallback policies (`CPU_HEAP`, `MMAP_WEIGHTS`, `GPU_PREFERRED`).
  - `LogicalDType`: Semantic numeric types separate from physical encoding.
  - `PackedBlockStorage`: Unified contract for all packed quantized formats.
  - `MemoryPlanner`, `MemoryTracker`, `ActiveMemoryTracker`: Placement resolution and copy diagnostics.
- **KV-Cache Subsystem**: `KvCacheStore` interface with append-by-token writes, layer/head addressing, eviction, and `DefaultKvCacheStore` (dense FP32 baseline).
- **Quantization-Preserving Loaders**: `StreamingGGUFReader` and `StreamingSafeTensorsReader` produce `TensorStorage` with `FileBacked` or `Borrowed` handles (no forced densification).
  - `StorageAwareSafeTensorsLoader`: Zero-copy file-backed SafeTensors loading.
  - Completed `Quants.kt` port: `byteShapeToQuantShape`, `quantByteSize`, `isBlockQuantized`, `validateQuantizedBytes`.
- **Tekken Tokenizer**: Mistral Tekken (tiktoken-based BPE) tokenizer support.
- **CPU SIMD TurboQuant Kernels**: `JvmTurboQuantKernels` with Java Vector API acceleration for abs-max, quantize, dequantize, and Walsh-Hadamard butterfly.
- **JMH Benchmarks**: TurboQuant encode/decode throughput, bit-packing, rotation, and KV cache append/read benchmarks (`TurboQuantBenchmarks.kt`).
- **Storage Benchmarks**: Dequantization throughput (Q4_K, Q8_0, Ternary), buffer accessor, and TensorData bridge benchmarks (`StorageBenchmarks.kt`).
- **New Ops**: `sin`, `cos`, `tanh`, `convTranspose1d`.
- **New Layers**: `TransposedConv1d`, `Snake` activation, `LayerScale`.

### Changed
- **Streaming GGUF as Default**: `StreamingGGUFReader` is now the recommended GGUF loading path (memory-efficient, supports quantized types).
- **DSL Annotations**: Extended `PlacementAnnotations.kt` with `@KvCache(preset=...)` and `@KvCacheBypass` for TurboQuant configuration.

### Fixed
- **Int Overflow for Large Tensors**: Fixed `StreamingTensorInfo.nBytes` and `StreamingSafeTensorInfo.sizeInBytes` from `Int` to `Long`, preventing silent overflow for tensors > 2 GB. Fixes loading of Gemma 4 E4B and future large models. (#452)
- **Legacy GGUFReader Overflow Guard**: Added explicit overflow check with actionable error message for tensors > 2 GB in the legacy eager loader.

### Dependencies
- io.github.kotest:kotest: 6.1.9 → 6.1.11.
- com.squareup:kotlinpoet: 2.2.0 → 2.3.0.

## [0.17.0] - 2026-03-25

### Added
- **Core Engine Focus**: Refactored the repository to focus on the core `ComputeGraph` framework, compiler, and backends.
- **Standalone Ecosystem**: Extracted high-level LLM and transformer implementations to dedicated repositories ([SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) and [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers)).
- **LLM-as-DSL**: High-level DSL for defining and running LLM architectures within the core `ComputeGraph` framework.
- **ComputeGraphExecutor**: New optimized executor with support for fusion passes and trace-to-DAG bridging.
- **SDPA & Gather**: Implementation of Scaled Dot-Product Attention (SDPA) and `gather`/`indexSelect` ops across backends.
- **EmbeddingAdapter**: Streamlined embedding layer integration for transformer models.

### Changed
- **Optimized LLM execution**: Integrated fusion passes for faster inference on supported backends.
- **Improved Tensor API**: Refined `Tensor` interface and updated `ComputeGraphExecutor` for better type safety and performance.
- **Dependency Cleanups**: Removed stale references to LLM and transformer code already moved to the standalone `skainet-transformers` repository.

### Fixed
- **Embedding Padding**: Fixed `paddingIdx` handling in embedding layers.
- **Concatenation**: Resolved rank-specific issues in tensor concatenation (rank > 1).
- **Compilation**: Fixed various build and compilation errors after module migrations.

## [0.16.0] - 2026-03-08

### Added
- **Deduplicated LLM infrastructure**: unified `KvCache`, `softmax`, `RoPE`, and `sampling` logic across modules for improved maintainability.
- **Updated skainet-bom**: Refactored the Bill of Materials (BOM) to use local `project()` references for better build consistency.

### Changed
- **LLM Module Extraction**: Extracted and moved core LLM modules to the standalone [SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) repository to reduce core codebase footprint.
- **Transformer Code Cleanup**: Removed redundant code that has been moved to the [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers) repository.

### Fixed
- **Dependency Graph**: Resolved inverted dependency issues in the LLM infrastructure.

## [0.15.3] - 2026-03-07

### Added
- **System Prompt Support (Java)**: Added `systemPrompt` support to `KLlamaJava` and `KLlamaSession` for prepending system instructions to conversations.
- **Model Module Extraction**: Extracted model-specific code into dedicated `skainet-models` modules for better separation of concerns and maintainability.
- **Enhanced Smoke Tests**: Refactored `smoke-test.sh` to support multiple runners via JSON configuration and improved LLM loading verification.

### Fixed
- **Whisper HLO Generation**: Fixed StableHLO MLIR generation for Whisper models.
- **Compilation**: Fixed various Kotlin/JVM compilation errors.

## [0.14.0] - 2026-03-03

### Added
- **First-Class Java 21+ Support**: Complete Java API surface with `SKaiNET` entry point, `TensorJavaOps`, builder-pattern model definition (`SequentialModelBuilder`), `KLlamaJava`/`KBertJava` facades, `JavaAgentLoop` for tool-calling agents, and `TrainingLoop` builder.
- **Maven BOM**: New `sk.ainet:skainet-bom` artifact for one-line version management across all modules.
- **Java Documentation**: Added [Getting Started](docs/java-getting-started.md), [LLM Inference](docs/java-llm-inference.md), and [Model Training](docs/java-model-training.md) guides.
- **Java 25 Performance Documentation**: Added documentation for JVM CPU backend performance advantages.
- **WasmWasi Target**: Added `wasmWasi` target support across all KMP modules.
- **StableHLO MLIR Streaming API**: New `HloGenerator` public API with generic Model + Tensor interface and streaming MLIR output.
- **ReductionOperationsConverter**: Added support for reduction operations in StableHLO export.
- **JVM Performance (Jlama Techniques)**: MemorySegment-based tensors, SIMD GEMM kernels, paged KV cache, batch attention for prompt prefill, fused QKV projections, and cached quantized weights.
- **Native RandomAccessSource**: POSIX `pread()`-based source for memory-efficient GGUF parsing.
- **MemorySegment Weight Conversion**: New `NATIVE_OPTIMIZED` quant policy and `MemSegWeightConverter` pipeline with Arena lifecycle management.
- **Lazy Transpose**: Added lazy transpose for Q4/Q8 MemorySegment tensors and MemSeg FP32 transpose.
- **Java CLI App**: New Java-based KLlama CLI application.

### Changed
- **Android KMP Plugin Migration**: Migrated Android subprojects to `androidMultiplatformLibrary` plugin for AGP 9 compatibility.
- **Refactored Model Loading**: Extracted shared dequantization, registry, tensor naming, and decoder runtime into reusable components.
- **JDK Requirement Relaxed**: Allow JDK >= 21 instead of requiring exactly JDK 21.
- **Gradle Upgrade**: Updated to Gradle 9.3.1.
- **Kotlin Upgrade**: Bumped Kotlin from 2.2.21 to 2.3.10.
- **Kotlin Compile Testing**: Replaced abandoned `kotlin-compile-testing` with `kctfork` for Kotlin 2.3.0 compatibility.

### Fixed
- **StableHLO MLIR Export**: Fixed MLIR export to produce valid IREE-compilable output.
- **OOM in Dequantization Benchmark**: Fixed out-of-memory in `DEQUANTIZE_TO_FP32` E2E benchmark test.
- **Quantized MatMul**: Fixed block offset calculation in quantized matrix multiplication.
- **CI Stability**: Fixed AAPT2 daemon crashes and improved Android build stability.
- **Documentation CI**: Fixed workflow permissions for PR comments.
- **Deprecated API Usage**: Fixed `createTempDir()` deprecation in data-simple integration tests.

### Dependencies
- com.gradleup.shadow: 9.3.1 → 9.3.2.
- com.fasterxml.jackson.core:jackson-databind: 2.21.0 → 2.21.1.
- ch.qos.logback:logback-classic: 1.5.27 → 1.5.32.
- io.github.kotest:kotest: 6.1.3 → 6.1.4.
- org.jetbrains.kotlinx:kotlinx-io-core: 0.8.2 → 0.9.0.
- com.vanniktech.maven.publish: → 0.36.0.
- org.jetbrains.kotlinx.kover: → 0.9.7.
- actions/setup-node: 4 → 6.
- actions/upload-artifact: 6 → 7.
- actions/download-artifact: 7 → 8.
- junit-platform-launcher added for CI test execution.

### Contributors
Thank you to the following contributors for their work on this release:
- **Dhia Chemingui** ([@dhiaspaner](https://github.com/dhiaspaner)) — Android KMP plugin migration (#385, #386)

## [0.13.0] - 2026-02-12

### Added
- **Tool Calling**: Added support for tool calling in KLlama, including a new `skainet-kllama-agent` module.
- **Gemma 3n Support**: New `skainet-kgemma` module for Google's Gemma 3n E2B multimodal models.
- **Extended SafeTensors Support**: Added SafeTensors weight loading support for both KLlama CLI and Gemma models.
- **HuggingFace Tokenizer**: Initial support for HuggingFace-style tokenizers in Gemma models.

### Changed
- **Named Arguments**: Refactored various internal APIs to use named arguments for better optional parameter support.
- **System Prompt Handling**: Improved system prompt formatting and handling in agentic workflows.

## [0.12.0] - 2026-02-10

### Added
- **BERT Support**: Full support for BERT-based models with `SafeTensors` weight loading.
- **kbert-cli**: New CLI tool for running BERT inference, supporting text encoding and cosine similarity computation.
- **WordPiece Tokenizer**: Implementation of WordPiece tokenizer for BERT models.

## [0.11.0] - 2026-02-08

### Added
- **TinyFoA Support**: Implemented missing operators (`abs`, `sign`, `clamp`, `lt`, `ge`, `narrow`, `pad2d`, `unfold`) to support TinyFoA (AAAI 2025) training pipeline for memory-efficient on-device learning.
- **Multi-platform KLlama**: Added macOS target support for the KLlama runtime.
- **Custom Backends Documentation**: Added detailed guide and examples for injecting custom backends into KLlama.

### Fixed
- Improved robustness of TinyFoA operations with comprehensive unit tests.

## [0.10.1] - 2026-02-01

### Added
- **Benchmarking DSL**: New `BenchmarkDsl` and `BenchmarkRunner` for measuring model performance and latency.
- **Execution Observers**: Added `ExecutionObserver` API with `LatencyExecutionObserver` and `MemorySnapshotObserver` for profiling.
- **New Layers**: Added `RMSNormalization` layer support.
- **KLlama Enhancements**: Improved weight loading and initial support for GPU-accelerated attention (experimental).

### Changed
- Refactored `ExecutionContext` to support execution observers and better phase management.
- Updated KLlama runtime with improved ingestion and benchmarking utilities.

## [0.9.2] - 2026-01-27

### Added
- **Generative AI Section**: New README section with simple code for GGUF text generation.
- **Tokenizer Strategies**: Automatic detection of tokenizer type (SentencePiece, BPE, WordPiece) from GGUF metadata.
- **Improved Token Decoding**: Support for multi-byte UTF-8 character decoding from byte tokens.

### Changed
- **Llama Runtime**: Rewritten `matmulNoBias` for better performance and support for row-major weights.
- **GGUF Loading**: Improved dequantization for Q2_K, Q4_K, Q5_K, and Q6_K formats matching llama.cpp logic.

### Fixed
- **GGUF Storage Order**: Fixed critical bug with column-major storage in GGUF files by implementing proper transposition during loading.
- **Llama Attention**: Fixed missing attention output projection (wo) in the runtime.
- **Tokenizer**: Fixed BOS token handling and multi-byte character reconstruction.

## [0.9.1] - 2026-01-26

### Added
- **SafeTensors Support**: Initial implementation of `skainet-io-safetensors` for reading SafeTensors format.
- **Generalized I/O & Weight Mapping**:
    - New `WeightMapper` and `WeightLoader` APIs for unified model parameter loading across formats.
    - `LoadingProgress` API for tracking model loading state.
    - `GgufModelMetadata` and `OnnxModelMetadata` for better inspection of model files.
- **JVM Performance**: Enhanced `DefaultCpuOpsJvm` with `JvmVectorKernels` for SIMD-accelerated tensor operations using the Java Vector API.
- **Llama Enhancements**:
    - Added `GGUFTokenizer` for better text processing.
    - Improved `LlamaIngestion` and ingestion pipelines.

### Changed
- **Improved GGUF/ONNX Loading**: Robust weight loading and metadata parsing for GGUF and ONNX models.
- **Streamlined CLI**: Removed unfinished CLI samples and reorganized `skainet-tensor-tools`.
- **Documentation Cleanup**: Removed outdated technical docs and consolidated architecture information.

### Fixed
- Improved robustness of GGUF and ONNX streaming readers.
- Fixed various issues in WASM/JS weight parsing.

## [0.8.3] - 2026-01-18

### Changed
- Updated version to 0.8.3.

## [0.8.2] - 2026-01-18

### Added
- **KLlama (Llama 2 port)**: Initial version ported from `llama2-kmp`, supporting GGUF models.
- **GGUF Enhancements**:
    - Support for `mmap` for zero-copy GGUF tensor loading.
    - Embedded tokenizer support in GGUF.
    - New quantization formats: `Q8_0`, `Q4_K`, and BitNet/Ternary support (`TQ1_0`, `TQ2_0`).
    - Improved loading and bug fixes for quantization and mapping.
    - Added `int64` support for GGUF.
    - Improved GGUF metadata loading.
- **Streaming Support**: Added streaming support for GGUF and ONNX models.
- **Advanced Operations**:
    - New activations: `LeakyReLU`, `ELU`.
    - New pooling: `AvgPool2d`.
    - New convolutions: `Conv1d`, `Conv3d`.
- **Optimizers & Training**:
    - Added `Adam` and `AdamW` optimizers.
    - Comprehensive loss function library.
    - New `Metric` interface with `Accuracy` implementation.
    - KSP-based DSL generator for Network activations.
- **Data & Datasets**:
    - Support for `CIFAR-10` and `Fashion-MNIST` datasets.
    - New `Data Transform API` and `Image Transform DSL`.
- **Testing & Documentation**:
    - `skainet-test-groundtruth` module for validation against PyTorch.
    - Integration tests for quantized inference and `KvCache`.
    - Shadow JAR support for JVM fat JAR builds.
    - New documentation for testing architecture with Mermaid diagrams.
- **WASM/JS**: Initial version of a simple WASM/JS sample.

### Changed
- Simplified model support to **GGUF-only** (removed legacy Karpathy `.bin` format support).
- Improved KLlama loading and robustness.
- Updated roadmap with Phase 1 completion and multi-backend storage abstraction plans.
- Improved I/O system and overall robustness.

### Fixed
- Fixed various bugs in quantization and memory mapping.
- Resolved compilation errors and failing tests in CIFAR-10 support.
- Fixed KSP and TracingWrapperProcessor tests to match updated log messages.
- Fixed GGUF metadata loading issues.

## [0.8.1] - 2026-01-18
- Initial release of 0.8.x series.

## [0.7.1] - 2026-01-14

### Added
- **Sine Approximation CLI** (`skainet-sine-approx-cli`) as a new example application for training models.
- `TapeRecordingStrategy` to handle different recording behaviors for prediction and backpropagation.
- Comprehensive E2E tests for training sine wave approximations.
- New documentation: `autograd-basic.md` explaining the autograd engine.

### Changed
- Refined `Linear`, `Flatten`, `Input` modules and `relu` activation to better support gradient tracking and context propagation.
- Improved `DefaultExecutionTape` and `DefaultGraphExecutionContext` for more robust computation tracing.
- Optimized internal `OpSink` and `TraceSession` handling.

### Fixed
- Infinite loop error during backpropagation tracing by implementing specialized tape recording strategies.
- Context mismatch errors in backpropagation tracing.
- Broken testing in the sinus sample application.

## [0.7.0] - 2026-01-14

### Added
- Initial **Autograd** engine (`DefaultGradientTape`) for automatic differentiation and reverse-mode gradients.
- **Optimizer API** with `SgdOptimizer` implementation for training neural networks.
- **Loss functions** module including `MSELoss` and `CrossEntropyLoss` with configurable reductions (MEAN, SUM, NONE).
- **Training DSL** and helper utilities for building training loops (`trainStep`, `evaluateLoss`).
- Improved **Graph DSL** with better context propagation and support for recording computation traces.

### Changed
- Updated dependency versions and refined internal execution context APIs to support gradient tracking.
- Refactored `skainet-compile-dag` to support autograd and graph inversion.

## [0.6.0] - 2025-12-31

### Added
- StableHLO implementation and E2E CLI app for compiling models to CUDA via IREE.
- `ArduinoCodegen` for exporting models to standalone C99 code with static memory allocation, optimized for Arduino.
- KSP-based generation of `TracingOps` for automated recording pipeline updates.
- Initial implementation of `skainet-compile-hlo` for high-level optimization.

### Changed
- Improved CUDA backend strategy and IREE integration.
- Optimized long-running property tests for C code generation.
- Refactored `TracingTensorOps` to use execution context for code generation.

## [0.5.1] - 2025-12-26

### Added
- Common I/O abstraction with `ModelReader` and `TensorInfo` in `skainet-io-core` for unified model loading.
- Efficient memory handling with non-copying `slice` views in `MemoryChunk`.
- Unified `skainet-tensor-tools` CLI combining ONNX and GGUF utilities.
- `OnnxStatsCli` tool for analyzing ONNX model parameters and structure.

### Changed
- Migrated project to `SKaiNET-developers` organization; updated repository URLs and deployment configurations.
- Standardized artifact naming in documentation (e.g., `SKaiNET-lang-core`).
- Improved `GGUFReader` with better alignment parsing and tensor data handling.
- Optimized test infrastructure: increased heap size to 8GB for large model tests and added `ReadmeSnippetsTest` for documentation verification.

### Removed
- Legacy standalone applications and tools: `skainet-KGPChat`, `skainet-mnist`, and separate ONNX/GGUF tool modules.

## [0.5.0] - 2025-12-06

### Added
- ONNX import module (`skainet-io-onnx`) with pbandk-generated proto surface, loader utilities, and importer that maps ONNX graphs into SKaiNET compute graphs, plus doc and tests.
- CLI tooling: `skainet-onnx-tools` to export ONNX initializers to JSON and `skainet-onnx-detect` CLI to run YOLO detections from ONNX weights.
- YOLOv8 model upgrades: depth/width scaling, decoupled heads with DFL projection, class-name parsing, and detection helpers to align with ONNX exports.
- Image IO module now published with explicit API surface for bitmap <-> tensor conversions across platforms.

### Changed
- BatchNorm now reshapes stats for broadcasting and exercises JVM/native tests; CPU backend implements `sqrt` to support it.

### Dependencies
- Added pbandk runtime 0.16.0 for ONNX protobuf decoding.

## [0.4.0] - 2025-12-03

### Added
- Recording/tracing pipeline for tensor ops (RecordingExecution/TracingTensorOps) and compute-graph DAG under `sk.ainet.lang.graph`, including tape-to-graph conversion and GraphViz export helpers/tests.
- JSON export proof of concept via new `skainet-compile-json` module with serialization models, `exportJson` CLI, and tiny graph golden fixtures.
- Multiplatform image IO module to convert platform bitmaps <-> tensors and RGB byte arrays; includes macOS implementation fixes.
- Dedicated YOLOv8 model module (`skainet-models:skainet-model-yolo`) with graph assembly, config/pre/post-processing, and missing upsample/concat ops required by the model.
- NN DSL additions: multi-input `Functional` wrapper, new `Upsample2d`/Softmax helpers, scalar DSL builder plus tensor/number operator overloads, and extra tensor view/pprint utilities.

### Changed
- Graph DSL relocated into the lang namespace with refreshed default execution tape/graph context wiring; removed unused integration module scaffolding.
- Removed committed MNIST training assets; rely on download at runtime.
- Added scalar arithmetic support across backends and void ops to match new operator overloads.

### Fixed
- Corrected unsqueeze view handling and data DSL dtype reuse; stabilized tracing/JSON/tape tests.
- Fixed macOS image conversion path and cleaned duplicate files in the new IO/image pipeline.

### Dependencies
- io.ktor client 3.3.3 (from 3.3.2).
- logback-classic 1.5.21 (from 1.5.20).

## [0.3.0] - 2025-11-27

### Added
- Kolmogorov–Arnold Network (KAN/AKN) module and DSL support, including public factory and aliases for direct construction. Introduces `Akn`/`AknConfig` and `createAkn` mirroring DSL defaults.
- Example KAN models and graphs (e.g., Sine function examples and pretrained variant) with tests and Graphviz export.
- Additional NN DSL conveniences around initialization scopes (weights/basis/bias) and activation hooks used by KAN.

### Changed
- Minor API refinements in lang/nn DSL to better align with execution context usage for new KAN modules.

### Fixed
- Stabilized integration tests for KAN modules and examples.

### Performance
- Minor initialization performance tweaks for new modules.

### Docs
- Updated docs and samples to include KAN usage and references.



## [0.2.0] - 2025-11-16

### Added
- Initial support for model code sharing API (model definition, execution, loading). Implements #196, related to #169.
- Batch Normalization layer. Implements #193.
- Forward hooks and simple tape recording for NN. Implements #190, related to #104.
- Common traversal base for modules, with tests; Embedding implementation with dual value types; switched EEmbeddings to DualModule implementation.
- Dropout (initial implementation) and phase support (training/eval) in execution context so modules can behave differently by phase. Related to #5.
- `tril` op (initial version).
- MaxPool op with DSL support; Conv2D DSL support.
- Data API: initial version including MNIST data loader; JSON loading support (renamed loader classes from CSV to JSON) with tests. Implements #180, #181; related to #176, #179.
- GGUF model loading implementation (initial import and working version). Implements #178, #182; related to #176, #177.
- MatMul support in backends.
- Nested data blocks support in DSL (data block returns a tensor); contexts for creating and collecting tensors (returning last or all created tensors).
- JVM Ops using the Java Vector API (initial implementation) and SIMD Vector API acceleration.
- JMH benchmarks (JVM module) and additional benchmarks.
- Sample showing general tensor calculations (e.g., image color transformations).

### Changed
- NN DSL refactored to use `ExecutionContext`; added `ExecutionContext` parameter to `forward` functions.
- Models and data APIs improved; unified tensor value creation in DSL; moved tensor creation context for safer vector/matrix/tensor creation.
- Default CPU compute used for JS target.
- JS and WASM Kotlin targets aligned for library packaging.
- Gradle updated to 9.0.0; Android target namespaces fixed.

### Fixed
- Crash in schema validation task; added Kotlin compiler plugin configuration for expect/actual.
- Activation not applied in Dense layer (fixed).
- JVM target issues; fixed failing JVM tests; added regression tests; stabilized platform matching test (temporarily ignored) and additional general test fixes.
- Miscellaneous build-signing validation added to avoid CI failures.

### Performance
- SIMD/Java Vector API acceleration for JVM backend operations.

### Dependencies
- com.vanniktech.maven.publish: 0.34.0 → 0.35.0.
- io.ktor (android, cio, content-negotiation, core, darwin, js, logging): 3.3.1 → 3.3.2.
- com.fasterxml.jackson.core:jackson-databind: 2.15.2 → 2.20.0 → 2.20.1.

### Build & CI
- GitHub Actions: use Java 22.
- Bump actions/checkout from v4 to v5.
- Add Gradle local caches to .gitignore.
- Preparations for 0.2.0 release and ability to build local Maven version of the upcoming release.

### Docs
- Added hint/reference on normalization layer paper. Related to #192.


## [0.1.0] - 2025-10-31
- Initial public release of SKaiNET 0.1.0.
