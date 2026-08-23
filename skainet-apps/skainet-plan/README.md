# skainet-plan — know before you load

`skainet plan` answers *will this model fit on this device at this context length?* from a GGUF
**header alone** (shapes, encodings and architecture metadata — no tensor bytes are read). It is
the milestone-M0 sample of the SKaiNET memory architecture (SKEEP-003, #1001/#1013).

```
$ ./gradlew :skainet-apps:skainet-plan:run --args="Llama-3.2-1B-Instruct-Q4_K_M.gguf --ctx 2048 --budget 1.3G"
Llama-3.2-1B-Instruct · llama · 16 layers · ctx 2048
  weights   Mapped, packed                  762 MB   resident
  kv cache  bf16 @ ctx 2048                  64 MB   resident   (17 MB with TurboQuant 4-bit)
  forward   prefill chunk 256                47 MB
  heap      headroom                         64 MB
  total                                     938 MB   of 1.3 GB  ✔ fits

$ … --budget 0.9G
  total                                     938 MB   of 921 MB  ✘ does not fit
  suggestions: --kv turboquant (−47 MB) · --ctx 1024 (−45 MB) · a smaller model: weights must shrink by ≥ 17 MB (…)

$ … --list 'model.layers[3].*'
  model.layers[3].attn.q_proj.weight   Float32/Q4_K      n=4194304       2 MB   ← blk.3.attn_q.weight
  model.layers[3].attn.k_proj.weight   Float32/Q4_K      n=1048576     576 KB   ← blk.3.attn_k.weight
  …
```

Options: `--ctx N` (default: the model's trained context length, else 2048) · `--budget 1.3G|900M|bytes`
(default: the JVM's max heap minus the 700 MB reserve of the 2 GB device profile) · `--kv bf16|turboquant` ·
`--prefill-chunk N` (default 256) · `--list <glob>` over `TensorId`s · `--no-budget`. Exit code 1 when the
plan does not fit the budget.

The planner itself (`sk.ainet.lang.memory.plan`, `StreamingGGUFReader.planInput`) is `commonMain`
code usable from Android and Kotlin/Native directly; this module is the JVM command line. The
numbers are estimates documented in `MemoryPlans` and calibrated by milestone M1's plan-vs-actual
check; unmapped tensor names (unknown architectures) are listed, never dropped.
