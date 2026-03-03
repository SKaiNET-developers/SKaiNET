# Changelog

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
