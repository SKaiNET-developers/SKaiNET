# SKaiNET LLM Extraction & Rework Plan

## 1. Current Architecture

```mermaid
graph TD
    subgraph SKaiNET Monorepo
        LANG[skainet-lang<br/>Tensor ops, layers, activations]
        COMPILE[skainet-compile<br/>Graph compilation, HLO, optimization]
        BACKENDS[skainet-backends<br/>CPU/GPU execution]
        IO[skainet-io<br/>GGUF, SafeTensors, ONNX, image]
        DATA[skainet-data<br/>Data loading, transforms]
        PIPELINE[skainet-pipeline]

        subgraph skainet-models
            LLAMA_M[skainet-model-llama<br/>LlamaRuntime, AttentionBackend interface,<br/>WeightLoader, ConfigParser, Quantization]
            GEMMA_M[skainet-model-gemma<br/>Gemma3nRuntime, Gemma3nKvCache,<br/>Hybrid attention, VisionEncoder]
            BERT_M[skainet-model-bert<br/>BertRuntime, HuggingFaceTokenizer,<br/>WordPiece, Embeddings]
            QWEN_M[skainet-model-qwen<br/>QwenConfigParser,<br/>HF tensor name mapper]
            YOLO_M[skainet-model-yolo<br/>Yolo8, Detection, NMS]
        end

        subgraph skainet-apps
            LLM_A[skainet-llm<br/>DecoderRuntime, Tokenizer interface,<br/>SentencePiece/BPE/WordPiece strategies]
            KLLAMA_A[skainet-kllama<br/>CpuAttentionBackend, GpuAttentionBackend,<br/>KvCache, GGUFTokenizer, LlamaIngestion]
            AGENT_A[skainet-kllama-agent<br/>AgentLoop, ChatTemplate, ToolRegistry,<br/>ToolCallParser, InferenceRuntime]
            KGEMMA_A[skainet-kgemma<br/>Gemma3nIngestion]
            CLI_A[skainet-kllama-cli<br/>skainet-kbert-cli]
            OTHER_A[skainet-grayscale-cli<br/>skainet-tensor-tools]
        end
    end

    LANG --> COMPILE
    LANG --> BACKENDS
    LANG --> IO
    COMPILE --> BACKENDS
    IO --> LLAMA_M
    IO --> GEMMA_M
    IO --> BERT_M
    LANG --> LLAMA_M
    LANG --> GEMMA_M
    LANG --> BERT_M
    LANG --> QWEN_M
    LANG --> YOLO_M
    LANG --> LLM_A
    LLM_A --> LLAMA_M
    LLM_A --> GEMMA_M
    LLM_A --> BERT_M
    AGENT_A --> KLLAMA_A
    LLM_A --> KLLAMA_A
    LLAMA_M --> KLLAMA_A
    QWEN_M --> LLAMA_M
    KLLAMA_A --> CLI_A
    KGEMMA_A --> GEMMA_M

    style LLAMA_M fill:#f9a825
    style GEMMA_M fill:#f9a825
    style BERT_M fill:#f9a825
    style QWEN_M fill:#f9a825
    style LLM_A fill:#f9a825
    style KLLAMA_A fill:#f9a825
    style AGENT_A fill:#f9a825
    style KGEMMA_A fill:#f9a825
```

**Yellow = LLM-related code to be extracted.**

---

## 2. Identified Duplications

### 2.1 KvCache Interface — HIGH severity

Two nearly identical interfaces with identical method signatures (`store`, `getKey`, `getValue`, `reset`) and identical fields (`nLayers`, `seqLen`, `kvDim`).

| | skainet-apps | skainet-models |
|---|---|---|
| **Interface** | `KvCache` in `skainet-kllama/…/KvCache.kt:13-61` | `Gemma3nKvCache` in `skainet-model-gemma/…/Gemma3nKvCache.kt:10-58` |
| **Heap impl** | `HeapKvCache` (lines 69-111) | `HeapGemma3nKvCache` (lines 72-150) |
| **Difference** | None | Adds `getCacheLayerIndex()` for layer sharing |

```mermaid
classDiagram
    class KvCache_apps {
        <<interface>>
        +nLayers: Int
        +seqLen: Int
        +kvDim: Int
        +store(layerIdx, position, keys, keysOffset, values, valuesOffset)
        +getKey(layerIdx): FloatArray
        +getValue(layerIdx): FloatArray
        +reset()
    }

    class Gemma3nKvCache_models {
        <<interface>>
        +nLayers: Int
        +seqLen: Int
        +kvDim: Int
        +store(layerIdx, position, keys, keysOffset, values, valuesOffset)
        +getKey(layerIdx): FloatArray
        +getValue(layerIdx): FloatArray
        +reset()
    }

    class HeapKvCache {
        -keyCache: FloatArray
        -valueCache: FloatArray
    }

    class HeapGemma3nKvCache {
        -keyCache: FloatArray
        -valueCache: FloatArray
        -getCacheLayerIndex(layerIdx): Int
    }

    KvCache_apps <|.. HeapKvCache
    Gemma3nKvCache_models <|.. HeapGemma3nKvCache

    note for KvCache_apps "DUPLICATE — identical contract"
    note for Gemma3nKvCache_models "DUPLICATE — identical contract"
```

**Fix:** Unify into a single `KvCache` interface. Gemma extends it with layer-sharing support.

### 2.2 Softmax — HIGH severity

Three independent implementations of numerically-stable softmax (find max, subtract, exp, normalize):

| Location | File | Lines |
|---|---|---|
| CpuAttentionBackend | `skainet-apps/skainet-kllama/…/CpuAttentionBackend.kt` | 189-206 |
| DecoderRuntime.sample() | `skainet-apps/skainet-llm/…/DecoderRuntime.kt` | 153-165 |
| GenerateExtensions | `skainet-apps/skainet-kllama-agent/…/GenerateExtensions.kt` | 96-108 |

**Fix:** Extract a single `softmaxInPlace(values: FloatArray, length: Int)` utility.

### 2.3 Sampling Logic — MEDIUM severity

Two independent implementations of greedy + temperature-scaled sampling:

| Location | File | Lines |
|---|---|---|
| DecoderRuntime.sample() | `skainet-apps/skainet-llm/…/DecoderRuntime.kt` | 138-173 |
| sampleFromLogits() | `skainet-apps/skainet-kllama-agent/…/GenerateExtensions.kt` | 76-116 |

Both have identical greedy path (`temperature <= 1e-6f` -> argmax) and identical softmax-then-random path.

**Fix:** Single `Sampler` utility used by both.

### 2.4 WordPiece Tokenization — MEDIUM severity

Two implementations of greedy longest-match-first with `##` prefix (~85% identical):

| Location | File | Lines |
|---|---|---|
| HuggingFaceTokenizer | `skainet-models/skainet-model-bert/…/HuggingFaceTokenizer.kt` | 233-276 |
| GGUFTokenizer | `skainet-apps/skainet-kllama/…/GGUFTokenizer.kt` | 605-662 |

**Fix:** Extract shared `WordPieceEncoder` utility.

### 2.5 RoPE Frequency Computation — MEDIUM severity

Same formula `freq = pos / base^(2*pair/dim)` duplicated in three files:

| Location | File | Lines |
|---|---|---|
| CpuAttentionBackend | `skainet-apps/skainet-kllama/…/CpuAttentionBackend.kt` | 108-120 |
| GpuAttentionBackend | `skainet-apps/skainet-kllama/…/GpuAttentionBackend.kt` | 60-81 |
| Gemma3nAttentionBackend | `skainet-models/skainet-model-gemma/…/Gemma3nAttentionBackend.kt` | 114-167 |

**Fix:** Shared `RopeCalculator` with pre-computed frequency tables.

### 2.6 Inverted Dependency — Architectural Smell

`skainet-models` depends on `skainet-apps` (backwards):

- `skainet-model-llama/build.gradle.kts:51` -> `skainet-apps:skainet-llm` (for `DecoderRuntime`)
- `skainet-model-llama/build.gradle.kts:52` -> `skainet-apps:skainet-kllama-agent` (for `InferenceRuntime`)

Models (architecture definitions) should NOT depend on apps (application code).

```mermaid
graph LR
    subgraph "Current (WRONG)"
        M1[skainet-models] -->|depends on| A1[skainet-apps]
    end

    subgraph "Target (CORRECT)"
        CORE[llm-core<br/>DecoderRuntime, KvCache,<br/>Sampler, RoPE] --> M2[llm-inference<br/>LLaMA, Gemma, BERT]
        CORE --> AG[llm-agent<br/>AgentLoop, Tools]
        M2 --> APP[llm-apps<br/>CLIs, demos]
        AG --> APP
    end

    style M1 fill:#e53935,color:#fff
    style A1 fill:#e53935,color:#fff
    style CORE fill:#43a047,color:#fff
    style M2 fill:#43a047,color:#fff
    style AG fill:#43a047,color:#fff
    style APP fill:#43a047,color:#fff
```

---

## 3. Extraction Plan: SKaiNET-LLM

### 3.1 Target Project Structure

```mermaid
graph TD
    subgraph "SKaiNET (stays)"
        S_LANG[skainet-lang<br/>Tensor ops, layers]
        S_COMPILE[skainet-compile<br/>Graph compilation]
        S_BACKENDS[skainet-backends<br/>CPU/GPU backends]
        S_IO[skainet-io<br/>GGUF, SafeTensors]
        S_DATA[skainet-data<br/>Data pipeline]
        S_YOLO[skainet-model-yolo<br/>Object detection]
        S_OTHER[skainet-grayscale-cli<br/>Vision demos]
    end

    subgraph "SKaiNET-LLM (new project)"
        LLM_CORE[llm-core]
        LLM_INF[llm-inference]
        LLM_AGENT[llm-agent]
        LLM_TRAIN[llm-training]
        LLM_DIST[llm-distributed]
        LLM_EVAL[llm-eval]
    end

    S_LANG --> LLM_CORE
    S_COMPILE --> LLM_CORE
    S_BACKENDS --> LLM_CORE
    S_IO --> LLM_INF

    LLM_CORE --> LLM_INF
    LLM_CORE --> LLM_AGENT
    LLM_CORE --> LLM_TRAIN
    LLM_INF --> LLM_AGENT
    LLM_TRAIN --> LLM_DIST
    LLM_INF --> LLM_EVAL

    style LLM_CORE fill:#1565c0,color:#fff
    style LLM_INF fill:#1565c0,color:#fff
    style LLM_AGENT fill:#1565c0,color:#fff
    style LLM_TRAIN fill:#2e7d32,color:#fff
    style LLM_DIST fill:#2e7d32,color:#fff
    style LLM_EVAL fill:#2e7d32,color:#fff
```

**Blue = extracted from SKaiNET. Green = new functionality.**

### 3.2 Module Breakdown

#### `llm-core` — Shared LLM Primitives

```
llm-core/
├── tokenizer/          # Tokenizer interface + strategies (BPE, SentencePiece, WordPiece)
├── sampling/           # Greedy, temperature, top-k, top-p, beam search
├── kv-cache/           # KvCache interface + HeapKvCache, OffheapKvCache, PagedKvCache
├── rope/               # RoPE utilities (standard, NTK-scaled, YaRN, dual-frequency)
├── attention/          # AttentionBackend interface, softmax utilities
└── runtime/            # DecoderRuntime base class, InferenceRuntime interface
```

**Source mapping:**

| From | To |
|---|---|
| `skainet-apps/skainet-llm/…/DecoderRuntime.kt` | `llm-core/runtime/` |
| `skainet-apps/skainet-llm/…/Tokenizer.kt` | `llm-core/tokenizer/` |
| `skainet-apps/skainet-llm/…/tokenizer/*Strategy.kt` | `llm-core/tokenizer/` |
| `skainet-apps/skainet-kllama/…/KvCache.kt` | `llm-core/kv-cache/` |
| `skainet-apps/skainet-kllama-agent/…/InferenceRuntime.kt` | `llm-core/runtime/` |
| `skainet-apps/skainet-kllama-agent/…/GenerateExtensions.kt` | `llm-core/sampling/` (deduplicated) |
| `skainet-models/skainet-model-llama/…/AttentionBackend.kt` | `llm-core/attention/` |
| RoPE code (3 locations) | `llm-core/rope/` (unified) |
| Softmax code (3 locations) | `llm-core/attention/` (unified) |

#### `llm-inference` — Model Runtimes

```
llm-inference/
├── llama/              # LlamaRuntime, CpuAttentionBackend, GpuAttentionBackend
│   │                   # LlamaConfigParser, LlamaWeightLoader, GGUFTokenizer
│   │                   # QuantizedTensorFactory, GraphAccelerator
│   └── qwen/           # QwenConfigParser, QwenHfTensorNameMapper (thin layer)
├── gemma/              # Gemma3nRuntime, Gemma3nAttentionBackend
│   │                   # Gemma3nConfig, Gemma3nKvCache (extends KvCache)
│   └── multimodal/     # VisionEncoder, AudioEncoder
├── bert/               # BertRuntime, HuggingFaceTokenizer, BertIngestion
└── loader/             # LlamaIngestion, Gemma3nIngestion (unified loading facade)
```

**Source mapping:**

| From | To |
|---|---|
| `skainet-models/skainet-model-llama/…/*` | `llm-inference/llama/` |
| `skainet-apps/skainet-kllama/…/CpuAttentionBackend.kt` | `llm-inference/llama/` |
| `skainet-apps/skainet-kllama/…/GpuAttentionBackend.kt` | `llm-inference/llama/` |
| `skainet-apps/skainet-kllama/…/GGUFTokenizer.kt` | `llm-inference/llama/` |
| `skainet-apps/skainet-kllama/…/LlamaIngestion.kt` | `llm-inference/loader/` |
| `skainet-models/skainet-model-gemma/…/*` | `llm-inference/gemma/` |
| `skainet-apps/skainet-kgemma/…/*` | `llm-inference/gemma/` |
| `skainet-models/skainet-model-bert/…/*` | `llm-inference/bert/` |
| `skainet-models/skainet-model-qwen/…/*` | `llm-inference/llama/qwen/` |

#### `llm-agent` — Agentic Framework

```
llm-agent/
├── chat/               # ChatTemplate, ChatMLTemplate, Llama3ChatTemplate, ChatMessage
├── tools/              # Tool, ToolDefinition, ToolCall, ToolCallParser, ToolRegistry
├── loop/               # AgentLoop, AgentConfig, AgentListener
└── memory/             # (future) Conversation memory, RAG integration
```

**Source:** all from `skainet-apps/skainet-kllama-agent/`

#### `llm-training` — NEW: Training Infrastructure

```
llm-training/
├── autograd/           # Reverse-mode autodiff on tensor graph
│   ├── GradTensor.kt           # Tensor wrapper tracking computation graph
│   ├── BackwardGraph.kt        # Backward pass graph builder
│   └── GradientAccumulator.kt  # Gradient aggregation
├── loss/               # Loss functions
│   ├── CrossEntropyLoss.kt     # Standard next-token prediction loss
│   └── LabelSmoothing.kt       # Regularization
├── optimizer/          # Optimizers
│   ├── AdamW.kt                # Default LLM optimizer
│   ├── SGD.kt                  # Baseline
│   └── LRScheduler.kt          # Cosine annealing, warmup, linear decay
├── data/               # Training data pipeline
│   ├── TokenizedDataset.kt     # Pre-tokenized dataset
│   ├── DataLoader.kt           # Batching, shuffling, prefetching
│   └── PackedSequences.kt      # Efficient packing of variable-length sequences
├── trainer/            # Training orchestration
│   ├── TrainingLoop.kt         # Forward -> loss -> backward -> step
│   ├── Checkpointing.kt        # Save/resume training state
│   └── MetricsTracker.kt       # Loss, perplexity, throughput logging
└── fine-tune/          # Parameter-efficient fine-tuning
    ├── LoRA.kt                 # Low-Rank Adaptation
    ├── QLoRA.kt                # Quantized LoRA
    └── AdapterLayer.kt         # Pluggable adapter pattern
```

#### `llm-distributed` — NEW: Distributed Training

```
llm-distributed/
├── communication/      # Collective operations
│   ├── AllReduce.kt            # Ring-AllReduce for gradient sync
│   ├── AllGather.kt            # Weight gathering for FSDP
│   ├── P2P.kt                  # Point-to-point for pipeline parallelism
│   └── Transport.kt            # gRPC / TCP / NCCL (via JNI) backends
├── strategy/           # Parallelism strategies
│   ├── DataParallel.kt         # Replicate model, split data
│   ├── PipelineParallel.kt     # Split layers across workers
│   ├── TensorParallel.kt       # Split within layers (attention heads, FFN columns)
│   └── HybridParallel.kt       # Combine DP + PP + TP
├── sharding/           # Memory optimization
│   ├── FSDP.kt                 # Fully Sharded Data Parallel
│   ├── ShardingPlan.kt         # Which params go where
│   └── GatherScatter.kt       # On-demand weight materialization
├── cluster/            # Cluster management
│   ├── WorkerRegistry.kt       # Discovery and health checks
│   ├── FaultTolerance.kt       # Elastic training, checkpoint recovery
│   └── K8sIntegration.kt       # Kubernetes operator support
└── mixed-precision/    # Precision management
    ├── FP16Training.kt         # FP16 forward/backward, FP32 master weights
    ├── BF16Training.kt         # BFloat16 support
    └── GradientScaling.kt      # Loss scaling to prevent underflow
```

#### `llm-eval` — NEW: Evaluation

```
llm-eval/
├── perplexity/         # Perplexity computation on validation sets
├── benchmarks/         # MMLU, HellaSwag, ARC, TruthfulQA runners
└── harness/            # Evaluation harness orchestration
```

---

## 4. Phased Execution Plan

```mermaid
gantt
    title SKaiNET-LLM Extraction & Build Plan
    dateFormat YYYY-MM-DD
    axisFormat %b %Y

    section Phase 0: Dedup
    Unify KvCache interface              :p0a, 2026-03-15, 3d
    Extract shared softmax utility       :p0b, 2026-03-15, 2d
    Extract shared RoPE utility          :p0c, 2026-03-17, 2d
    Unify sampling logic                 :p0d, 2026-03-17, 2d
    Extract WordPiece utility            :p0e, 2026-03-19, 2d
    Fix inverted dependency direction    :p0f, 2026-03-21, 3d

    section Phase 1: Extract
    Create SKaiNET-LLM repo              :p1a, 2026-03-24, 1d
    Move llm-core                        :p1b, 2026-03-25, 5d
    Move llm-inference (LLaMA + Qwen)    :p1c, 2026-03-30, 5d
    Move llm-inference (Gemma + BERT)    :p1d, 2026-04-04, 5d
    Move llm-agent                       :p1e, 2026-04-09, 3d
    Wire up Gradle composite build       :p1f, 2026-04-12, 3d
    Verify all tests pass                :p1g, 2026-04-15, 3d

    section Phase 2: Training
    Autograd (reverse-mode autodiff)     :p2a, 2026-04-18, 21d
    Loss functions (CrossEntropy)        :p2b, 2026-05-09, 5d
    AdamW optimizer                      :p2c, 2026-05-14, 7d
    LR schedulers (cosine, warmup)       :p2d, 2026-05-21, 5d
    Training loop + checkpointing        :p2e, 2026-05-26, 10d
    Data pipeline (tokenized datasets)   :p2f, 2026-06-05, 7d
    LoRA / QLoRA fine-tuning             :p2g, 2026-06-12, 10d
    Evaluation harness                   :p2h, 2026-06-22, 7d

    section Phase 3: Distributed
    Communication primitives (AllReduce) :p3a, 2026-06-29, 14d
    Data Parallel strategy               :p3b, 2026-07-13, 10d
    Gradient accumulation                :p3c, 2026-07-23, 5d
    Mixed precision (FP16/BF16)          :p3d, 2026-07-28, 10d
    Pipeline parallelism                 :p3e, 2026-08-07, 14d
    Tensor parallelism                   :p3f, 2026-08-21, 14d
    FSDP                                 :p3g, 2026-09-04, 14d
    Elastic training + fault tolerance   :p3h, 2026-09-18, 10d
```

---

## 5. Phase 0: Deduplication Details

### 5.1 Unify KvCache

```kotlin
// llm-core/kv-cache/KvCache.kt
public interface KvCache {
    public val nLayers: Int
    public val seqLen: Int
    public val kvDim: Int
    public fun store(layerIdx: Int, position: Int, keys: FloatArray, keysOffset: Int, values: FloatArray, valuesOffset: Int)
    public fun getKey(layerIdx: Int): FloatArray
    public fun getValue(layerIdx: Int): FloatArray
    public fun reset()
}

// llm-core/kv-cache/HeapKvCache.kt
public class HeapKvCache(override val nLayers: Int, override val seqLen: Int, override val kvDim: Int) : KvCache { ... }

// llm-inference/gemma/SharedLayerKvCache.kt  (extends KvCache with layer sharing)
public class SharedLayerKvCache(
    override val nLayers: Int, override val seqLen: Int, override val kvDim: Int,
    private val sharingStartLayer: Int
) : KvCache {
    private fun getCacheLayerIndex(layerIdx: Int): Int = ...
}
```

### 5.2 Unify Softmax + Sampling

```kotlin
// llm-core/sampling/SamplingUtils.kt
public fun softmaxInPlace(values: FloatArray, length: Int) { ... }

public fun sampleFromLogits(logits: FloatArray, temperature: Float, random: Random = Random.Default): Int { ... }
```

### 5.3 Unify RoPE

```kotlin
// llm-core/rope/RopeFrequencies.kt
public class RopeFrequencies(
    public val dim: Int,
    public val base: Float = 10000f,
    public val maxSeqLen: Int = 4096
) {
    private val cosTable: FloatArray = ...  // pre-computed
    private val sinTable: FloatArray = ...

    public fun cos(pair: Int, pos: Int): Float = cosTable[pos * (dim / 2) + pair]
    public fun sin(pair: Int, pos: Int): Float = sinTable[pos * (dim / 2) + pair]

    public fun applyRotary(q: FloatArray, k: FloatArray, pos: Int, nHeads: Int, nKvHeads: Int) { ... }
}
```

---

## 6. Technical Decisions for Distributed Training

| Decision | Recommendation | Rationale |
|---|---|---|
| **Communication** | gRPC + raw binary protocol | JVM-native; protobuf for control plane, raw ByteBuffer for tensor data |
| **Tensor transfer** | Off-heap `ByteBuffer` / `MemorySegment` | Avoid GC pressure on large gradient buffers |
| **GPU collective ops** | JNI to NCCL | NVIDIA AllReduce is 10-100x faster than custom TCP |
| **Cluster orchestration** | Kubernetes + custom operator | Standard for JVM services, supports auto-scaling and elastic training |
| **Checkpointing** | Async to object storage (S3/GCS) | Non-blocking, durable, resumable |
| **Fault tolerance** | Elastic training with checkpoint resume | Workers can join/leave without full restart |
| **Serialization** | Custom binary (not protobuf for tensors) | Protobuf adds overhead for large float arrays |

---

## 7. Dependency Graph After Extraction

```mermaid
graph TD
    subgraph "SKaiNET (general ML framework)"
        LANG[skainet-lang]
        COMPILE[skainet-compile]
        BACKENDS[skainet-backends]
        IO[skainet-io]
        DATA[skainet-data]
    end

    subgraph "SKaiNET-LLM (focused LLM project)"
        CORE[llm-core<br/>Tokenizer, KvCache, RoPE,<br/>Sampler, DecoderRuntime]
        INF[llm-inference<br/>LLaMA, Gemma, BERT, Qwen]
        AGENT[llm-agent<br/>AgentLoop, Tools, Chat]
        TRAIN[llm-training<br/>Autograd, AdamW, LoRA,<br/>TrainingLoop]
        DIST[llm-distributed<br/>DataParallel, FSDP,<br/>PipelineParallel, AllReduce]
        EVAL[llm-eval<br/>Perplexity, Benchmarks]
    end

    LANG --> CORE
    COMPILE --> CORE
    BACKENDS --> CORE
    IO --> INF
    DATA --> TRAIN

    CORE --> INF
    CORE --> AGENT
    CORE --> TRAIN
    INF --> AGENT
    INF --> EVAL
    TRAIN --> DIST
    TRAIN --> EVAL

    style CORE fill:#1565c0,color:#fff
    style INF fill:#1976d2,color:#fff
    style AGENT fill:#1976d2,color:#fff
    style TRAIN fill:#2e7d32,color:#fff
    style DIST fill:#388e3c,color:#fff
    style EVAL fill:#388e3c,color:#fff
```

---

## 8. Migration Strategy

### Gradle Composite Build (transitional)

During migration, use Gradle composite builds so SKaiNET and SKaiNET-LLM can coexist:

```kotlin
// SKaiNET/settings.gradle.kts (during transition)
includeBuild("../SKaiNET-LLM") {
    dependencySubstitution {
        substitute(module("sk.ainet.llm:llm-core")).using(project(":llm-core"))
        substitute(module("sk.ainet.llm:llm-inference")).using(project(":llm-inference"))
        substitute(module("sk.ainet.llm:llm-agent")).using(project(":llm-agent"))
    }
}
```

### What Stays in SKaiNET

- `skainet-lang/` — general tensor library (foundation for all ML, not just LLM)
- `skainet-compile/` — graph compilation
- `skainet-backends/` — CPU/GPU execution
- `skainet-io/` — format readers (GGUF, SafeTensors, ONNX, image)
- `skainet-data/` — data pipeline
- `skainet-models/skainet-model-yolo` — vision model (not LLM)
- `skainet-apps/skainet-grayscale-cli` — vision demo
- `skainet-apps/skainet-tensor-tools` — general model analysis tools
- `skainet-pipeline/` — general pipeline
- `skainet-test/` — test infrastructure

### What Moves to SKaiNET-LLM

| SKaiNET module | SKaiNET-LLM target | Notes |
|---|---|---|
| `skainet-apps/skainet-llm` | `llm-core` | DecoderRuntime, Tokenizer, strategies |
| `skainet-apps/skainet-kllama-agent` | `llm-agent` | AgentLoop, tools, chat templates |
| `skainet-apps/skainet-kllama` | `llm-inference/llama` | Attention backends, KvCache, tokenizer, ingestion |
| `skainet-apps/skainet-kllama-cli` | `llm-inference/llama` (or separate cli module) | CLI entry points |
| `skainet-apps/skainet-kgemma` | `llm-inference/gemma` | Gemma ingestion |
| `skainet-apps/skainet-kbert-cli` | `llm-inference/bert` | BERT CLI |
| `skainet-models/skainet-model-llama` | `llm-inference/llama` | Merge with kllama |
| `skainet-models/skainet-model-gemma` | `llm-inference/gemma` | Merge with kgemma |
| `skainet-models/skainet-model-bert` | `llm-inference/bert` | Merge with kbert |
| `skainet-models/skainet-model-qwen` | `llm-inference/llama/qwen` | Thin wrapper, fold into llama |

---

## 9. Distributed Training Architecture

```mermaid
graph TD
    subgraph "Cluster"
        subgraph "Worker 0 (Coordinator)"
            W0_MODEL[Model Shard 0]
            W0_OPT[Optimizer State 0]
            W0_GRAD[Gradient Buffer 0]
        end

        subgraph "Worker 1"
            W1_MODEL[Model Shard 1]
            W1_OPT[Optimizer State 1]
            W1_GRAD[Gradient Buffer 1]
        end

        subgraph "Worker N"
            WN_MODEL[Model Shard N]
            WN_OPT[Optimizer State N]
            WN_GRAD[Gradient Buffer N]
        end
    end

    DATA_SHARD[Data Sharding<br/>Each worker gets<br/>different mini-batch] --> W0_MODEL
    DATA_SHARD --> W1_MODEL
    DATA_SHARD --> WN_MODEL

    W0_GRAD <-->|Ring AllReduce<br/>gradient sync| W1_GRAD
    W1_GRAD <-->|Ring AllReduce| WN_GRAD
    WN_GRAD <-->|Ring AllReduce| W0_GRAD

    CKPT[Checkpoint Store<br/>S3 / GCS / NFS] -.->|async save| W0_MODEL
    CKPT -.->|async save| W1_MODEL
    CKPT -.->|async save| WN_MODEL

    style DATA_SHARD fill:#ff8f00,color:#fff
    style CKPT fill:#6a1b9a,color:#fff
```

### Parallelism Strategy Selection Guide

```mermaid
flowchart TD
    START[Model Size?] --> SMALL{< 10B params}
    START --> MEDIUM{10B - 100B params}
    START --> LARGE{> 100B params}

    SMALL --> DP[Data Parallel<br/>Full model replica per GPU]
    SMALL --> LORA[LoRA/QLoRA<br/>Fine-tune on single GPU]

    MEDIUM --> FSDP_M[FSDP<br/>Shard weights across GPUs]
    MEDIUM --> DP_GA[Data Parallel +<br/>Gradient Accumulation]

    LARGE --> HYBRID[Hybrid Parallelism<br/>TP + PP + DP]
    LARGE --> FSDP_L[FSDP +<br/>Activation Checkpointing]

    DP --> COMM1[AllReduce gradients<br/>after each step]
    FSDP_M --> COMM2[AllGather weights on demand<br/>ReduceScatter gradients]
    HYBRID --> COMM3[AllReduce within DP group<br/>P2P within PP stages<br/>AllReduce within TP group]

    style DP fill:#1565c0,color:#fff
    style LORA fill:#1565c0,color:#fff
    style FSDP_M fill:#f57f17,color:#fff
    style DP_GA fill:#f57f17,color:#fff
    style HYBRID fill:#c62828,color:#fff
    style FSDP_L fill:#c62828,color:#fff
```
