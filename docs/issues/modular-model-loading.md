# Modular LLM Model Loading Architecture

> **Epic:** Decouple model architectures from file formats so each model is a self-contained module that any format loader can feed into.

---

## Issue #1: Move format-independent abstractions to `skainet-io-core`

**Labels:** `refactor`, `io-core`, `phase-1`

`ModelArchitecture` and `QuantPolicy` are format-independent concepts that currently live in `skainet-io-gguf`. They should be in `skainet-io-core` so model modules can depend on them without pulling in GGUF.

### Subtasks

- [ ] Create `skainet-io-core/.../sk/ainet/io/model/ModelArchitecture.kt` with enum entries: `LLAMA`, `GEMMA`, `BERT`, `YOLO`
- [ ] Create `skainet-io-core/.../sk/ainet/io/model/QuantPolicy.kt`
- [ ] Replace `skainet-io-gguf/registry/ModelArchitecture.kt` with deprecated typealias pointing to `skainet-io-core`
- [ ] Replace `skainet-io-gguf/dequant/QuantPolicy.kt` with deprecated typealias pointing to `skainet-io-core`
- [ ] Keep GGUF-specific string→enum mapping (`"llama"` → `LLAMA`, `"gemma3n"` → `GEMMA`) as extension function in `skainet-io-gguf`
- [ ] Update all imports across the codebase
- [ ] `./gradlew build` passes
- [ ] `./gradlew allTests` passes

### Acceptance Criteria

- `ModelArchitecture` and `QuantPolicy` are owned by `skainet-io-core`
- `skainet-io-gguf` re-exports via `@Deprecated` typealias for backward compatibility
- No model-specific code in `skainet-io-core`

---

## Issue #2: Create `skainet-model-llama` module

**Labels:** `refactor`, `model-llama`, `phase-2`
**Depends on:** #1

Extract all Llama model logic into a self-contained `skainet-models/skainet-model-llama` module. This is the largest move — Llama code is currently split across `skainet-io-gguf/llama/` (weights, metadata, tensor names) and `skainet-apps/skainet-kllama/` (runtime, SafeTensors loader, config parser, JVM weight converter).

### Subtasks

#### Module setup
- [ ] Create `skainet-models/skainet-model-llama/build.gradle.kts` (KMP: JVM, iOS, macOS, Linux, Android, JS, WASM)
- [ ] Add dependencies: `skainet-lang-core`, `skainet-io-core`, `skainet-io-gguf`, `skainet-io-safetensors`, `skainet-backend-cpu`
- [ ] Register module in `settings.gradle.kts`

#### Move from `skainet-io-gguf/llama/` → `skainet-model-llama/commonMain/`
- [ ] `LlamaRuntimeWeights.kt` — runtime weight data classes
- [ ] `LlamaWeightLoader.kt` → rename to `LlamaGgufWeightSource.kt` — GGUF bridge
- [ ] `LlamaGgufTensorNames.kt` — GGUF tensor name mappings
- [ ] Leave `QuantizedTensorFactory.kt` in `skainet-io-gguf` (GGUF-specific quant handling)
- [ ] Leave `MmapLlamaLoader.kt` / `QuantizedTensorFactoryJvm.kt` in gguf or move to model-llama jvmMain if appropriate

#### Move from `skainet-apps/skainet-kllama/` → `skainet-model-llama/commonMain/`
- [ ] `LlamaRuntime.kt` — transformer forward pass (compute graph)
- [ ] `LlamaRuntimeInterface.kt` — runtime interface
- [ ] `LlamaSafeTensorsLoader.kt` → rename to `LlamaSafeTensorsWeightSource.kt`
- [ ] `LlamaConfigParser.kt` — HuggingFace config.json parsing

#### Move from `skainet-apps/skainet-kllama/` → `skainet-model-llama/jvmMain/`
- [ ] `MemSegWeightConverter.kt` — JVM off-heap zero-copy optimization

#### Update `skainet-kllama` to depend on `skainet-model-llama`
- [ ] Add `skainet-model-llama` dependency to `skainet-kllama/build.gradle.kts`
- [ ] Update all imports in `skainet-kllama` (KvCache, AttentionBackend, Tokenizer, Ingestion, CLI stay)
- [ ] Update all imports in `skainet-io-gguf` (registry, loader descriptors)

#### Keep in `skainet-kllama/` (app-level concerns)
- [ ] Verify `KvCache`, `AttentionBackend`, `CpuAttentionBackend`, `GpuAttentionBackend` stay
- [ ] Verify `GGUFTokenizer`, `TokenizerImpl`, `TokenizerUtils` stay
- [ ] Verify `LlamaIngestion`, `KLlamaJava`, CLI code stay

#### Tests
- [ ] Move relevant tests from `skainet-kllama` to `skainet-model-llama` (e.g., `LlamaRuntimeTest`, `MemSegWeightConverterTest`)
- [ ] `./gradlew build` passes
- [ ] `./gradlew allTests` passes
- [ ] End-to-end: load Q4_K_M model via `skainet-model-llama` and generate text

### Acceptance Criteria

- `skainet-model-llama` contains: runtime, weights, metadata, tensor names, config parser, GGUF loader, SafeTensors loader, JVM MemSeg converter
- `skainet-kllama` contains only: CLI, KvCache, tokenizer, attention backends, ingestion orchestration, Java API
- `skainet-io-gguf/llama/` is empty or contains only GGUF-specific helpers (`QuantizedTensorFactory`)

---

## Issue #3: Create `skainet-model-gemma` module

**Labels:** `refactor`, `model-gemma`, `phase-3`
**Depends on:** #1

Extract all Gemma 3n model logic into `skainet-models/skainet-model-gemma`. Gemma code is currently split across `skainet-io-gguf/gemma/` (weights, metadata, tensor names, config parser, SafeTensors loader) and `skainet-apps/skainet-kgemma/` (runtime, KV cache, attention backend, ingestion).

### Subtasks

#### Module setup
- [ ] Create `skainet-models/skainet-model-gemma/build.gradle.kts` (same KMP targets as model-llama)
- [ ] Add dependencies: `skainet-lang-core`, `skainet-io-core`, `skainet-io-gguf`, `skainet-io-safetensors`, `skainet-backend-cpu`
- [ ] Register module in `settings.gradle.kts`

#### Move from `skainet-io-gguf/gemma/` → `skainet-model-gemma/commonMain/`
- [ ] `Gemma3nModelMetadata.kt` — model metadata & layer type enum
- [ ] `Gemma3nRuntimeWeights.kt` — weight data classes, tensor names, weight mapper
- [ ] `Gemma3nWeightLoader.kt` → rename to `Gemma3nGgufWeightSource.kt`
- [ ] `Gemma3nSafeTensorsWeightLoader.kt` → rename to `Gemma3nSafeTensorsWeightSource.kt`
- [ ] `Gemma3nConfigParser.kt` — HuggingFace config.json parsing
- [ ] `Gemma3nGgufTensorNames.kt` — GGUF tensor name mappings

#### Move from `skainet-apps/skainet-kgemma/` → `skainet-model-gemma/commonMain/`
- [ ] `Gemma3nRuntime.kt` — transformer forward pass (GELU, hybrid attention, MatFormer FFN)
- [ ] `Gemma3nConfig.kt` — runtime configuration (sliding window, RoPE bases, KV sharing)
- [ ] `Gemma3nKvCache.kt` — KV cache with layer sharing
- [ ] `Gemma3nAttentionBackend.kt` + `AttentionBackend.kt` — hybrid attention (sliding + global)
- [ ] `multimodal/VisionEncoder.kt`, `multimodal/AudioEncoder.kt` — multimodal placeholders

#### Update `skainet-kgemma` to depend on `skainet-model-gemma`
- [ ] Add `skainet-model-gemma` dependency to `skainet-kgemma/build.gradle.kts`
- [ ] Update all imports in `skainet-kgemma`
- [ ] `skainet-kgemma` retains only: CLI entry points, ingestion orchestration

#### Keep in `skainet-kgemma/` (app-level concerns)
- [ ] Verify `Gemma3nIngestion.kt` stays (or decide if it moves)
- [ ] Verify CLI `Main.kt` (jvmMain, nativeMain) stays

#### Tests & verification
- [ ] `./gradlew build` passes
- [ ] `./gradlew allTests` passes
- [ ] End-to-end: load Gemma 3n GGUF and SafeTensors models, generate text

### Acceptance Criteria

- `skainet-model-gemma` contains: runtime, weights, metadata, config, KV cache, attention backend, tensor names, both loaders, multimodal encoders
- `skainet-kgemma` contains only: CLI and ingestion orchestration
- `skainet-io-gguf/gemma/` is empty

---

## Issue #4: Create `skainet-model-bert` module

**Labels:** `refactor`, `model-bert`, `phase-3`
**Depends on:** #1

BERT is already relatively well-contained in `skainet-apps/skainet-bert/` but should move to `skainet-models/skainet-model-bert/` for consistency. Unlike Llama and Gemma, BERT has no code in `skainet-io-gguf` (it loads only from SafeTensors). This is the simplest move.

### Subtasks

#### Module setup
- [ ] Create `skainet-models/skainet-model-bert/build.gradle.kts` (same KMP targets)
- [ ] Add dependencies: `skainet-lang-core`, `skainet-io-core`, `skainet-io-safetensors`, `skainet-backend-cpu`, `skainet-compile-core`
- [ ] Register module in `settings.gradle.kts`

#### Move from `skainet-apps/skainet-bert/` → `skainet-model-bert/`
- [ ] `BertRuntime.kt` — encoder forward pass (bidirectional attention, GELU, mean pooling)
- [ ] `BertRuntimeWeights.kt` — weight data classes, `BertModelConfig`
- [ ] `BertWeightLoader.kt` — tensor names, weight mapper, loading functions
- [ ] `BertIngestion.kt` — ingestion facade
- [ ] `HuggingFaceTokenizer.kt` — WordPiece tokenizer

#### Move JVM-specific code
- [ ] `java/KBertJava.kt` → `skainet-model-bert/jvmMain/`

#### Move tests
- [ ] `HuggingFaceTokenizerTest.kt` → `skainet-model-bert/commonTest/`
- [ ] `BertRuntimeTest.kt` → `skainet-model-bert/jvmTest/`
- [ ] `BertNumericalAccuracyTest.kt` → `skainet-model-bert/jvmTest/`

#### Update `skainet-kbert-cli` to depend on `skainet-model-bert`
- [ ] Add `skainet-model-bert` dependency to `skainet-kbert-cli/build.gradle.kts`
- [ ] Update imports in CLI `Main.kt` and `Demo.kt`
- [ ] Remove or deprecate `skainet-apps/skainet-bert/` module

#### Tests & verification
- [ ] `./gradlew build` passes
- [ ] `./gradlew allTests` passes

### Acceptance Criteria

- `skainet-model-bert` is a self-contained model module under `skainet-models/`
- `skainet-kbert-cli` depends on `skainet-model-bert` and contains only CLI code
- `skainet-apps/skainet-bert/` is removed

---

## Issue #5: Clean up deprecated typealiases and old file locations

**Labels:** `cleanup`, `phase-4`
**Depends on:** #2, #3, #4

Remove deprecated typealiases from `skainet-io-gguf` and delete emptied source directories.

### Subtasks

- [ ] Remove deprecated typealias for `ModelArchitecture` in `skainet-io-gguf`
- [ ] Remove deprecated typealias for `QuantPolicy` in `skainet-io-gguf`
- [ ] Delete `skainet-io-gguf/llama/` directory (after confirming only `QuantizedTensorFactory` remains, move if needed)
- [ ] Delete `skainet-io-gguf/gemma/` directory
- [ ] Remove moved files from `skainet-apps/skainet-kllama/` (LlamaRuntime, LlamaConfigParser, LlamaSafeTensorsLoader, MemSegWeightConverter)
- [ ] Remove moved files from `skainet-apps/skainet-kgemma/` (Gemma3nRuntime, config, KV cache, attention backend)
- [ ] Remove `skainet-apps/skainet-bert/` module entirely
- [ ] Update `settings.gradle.kts` to remove old module registrations
- [ ] Update `ModelLoaderRegistry.kt`, `LlamaLoaderDescriptor.kt`, `Gemma3nLoaderDescriptor.kt` in gguf registry to point to new locations
- [ ] Full search for any remaining imports to old packages
- [ ] `./gradlew build` passes
- [ ] `./gradlew allTests` passes

### Acceptance Criteria

- No deprecated typealiases remain
- No model-specific code in `skainet-io-gguf` (only format reading, dequant, export)
- No model runtime/weights code in `skainet-apps/` modules
- Clean module boundaries

---

## Issue #6: Prove modularity — add Qwen support

**Labels:** `feature`, `model-qwen`, `phase-5`
**Depends on:** #2, #5

Demonstrate the new architecture by adding Qwen2 support with minimal effort. Qwen2 uses the same transformer architecture as Llama, so it reuses `LlamaRuntime` and only needs its own config parser and tensor name mapper.

### Subtasks

- [ ] Create `skainet-models/skainet-model-qwen/build.gradle.kts`
- [ ] Add dependency on `skainet-model-llama`, `skainet-io-core`, `skainet-io-gguf`
- [ ] Register module in `settings.gradle.kts`
- [ ] Create `QwenConfigParser.kt` — parse Qwen2 HuggingFace config.json
- [ ] Create `QwenHfTensorNameMapper.kt` — map HuggingFace tensor names to canonical names
- [ ] Create `QwenGgufWeightSource.kt` — thin wrapper delegating to Llama weight loading with Qwen tensor names
- [ ] Add `QWEN` to `ModelArchitecture` enum in `skainet-io-core` (1 line)
- [ ] Add GGUF mapping `"qwen2" → QWEN` in `skainet-io-gguf` extension (1 line)
- [ ] **Verify:** zero changes to `skainet-io-gguf` format code, `skainet-model-llama`, or any app module
- [ ] `./gradlew build` passes
- [ ] End-to-end: load a Qwen2 GGUF model and generate text

### Acceptance Criteria

- Qwen2 model loads and runs inference
- Only new files are in `skainet-model-qwen/`
- Format modules and existing model modules are untouched (except 1-line enum + 1-line mapping)

---

## Summary

| Issue | Module | Phase | Effort | Depends on |
|-------|--------|-------|--------|------------|
| #1 | `skainet-io-core` | 1 | S | — |
| #2 | `skainet-model-llama` | 2 | XL | #1 |
| #3 | `skainet-model-gemma` | 3 | L | #1 |
| #4 | `skainet-model-bert` | 3 | M | #1 |
| #5 | cleanup | 4 | M | #2, #3, #4 |
| #6 | `skainet-model-qwen` | 5 | S | #2, #5 |

Issues #3 and #4 can be worked on in parallel after #1 is complete.
