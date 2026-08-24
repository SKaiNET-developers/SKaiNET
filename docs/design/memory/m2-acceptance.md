# SKEEP-003 M1 and M2 acceptance — what is measured, and where

Milestone record for [#1042](https://github.com/SKaiNET-developers/SKaiNET/issues/1042), closing
the M2 tracker [#1003](https://github.com/SKaiNET-developers/SKaiNET/issues/1003). Every row says
what was checked, where the number comes from, and — where a criterion is not closed — what is
missing. A criterion asserted by a test in this repository runs on every commit, on every target
the test suite covers.

The device numbers below come from an **ARMv8.2 Cortex-A55 reference board**: two cores,
1.9 GB RAM, `asimddp`, Linux. It is a 2 GB-class device, which is the class M2 targets, but it is
not Android — it has no ART, so the criteria that are about the *managed heap* cannot be closed
there.

## M1 — Flat decode

| ID | Criterion | Status | Evidence |
|---|---|---|---|
| M1-A1 | Memory flat across decode steps | **met** | `DecodeAcceptanceTest.m1a1…` — forward scope 0 bytes between steps, one distinct per-step value after warm-up. Runs in CI and on the reference board. |
| M1-A2 | Peak RSS during load ≤ file + KV + slab + 100 MB | open | needs a real GGUF and a load path with a model; belongs to the decode sample in SKaiNET-transformers |
| M1-A3 | Zero forward-scope allocations per step after warm-up | **met** | `DecodeAcceptanceTest.m1a3…` — 0 FORWARD allocations between steps 4→12 |
| M1-A4 | #993 / #991 through the registry | **met** | repro tests in #1027, dispatched by `KernelKey`, no special-casing |
| M1-A5 | Decode tok/s and matmul benchmarks within 3 % | partial | no hot path was modified (argued per PR); a tok/s number needs the real-model sample |
| M1-A6 | Packed matmul bit-identical for every encoding | **met** | golden parity gate (`scripts/pr-gate.sh --golden`), JVM + Kotlin/Native |
| M1-A7 | Perfetto trace shape | **met** | `DecodeAcceptanceTest.m1a7…` — one track per scope, kernel spans by `TensorId`, live-bytes counter returning to zero |
| M1-A8 | Plan vs actual within 10 % | **met** | `DecodeAcceptanceTest.m1a8…` — `PlanVsActual.withinTolerance`, adapter bytes 0 |
| M1-A9 | develop green, API source-compatible | **met** | every slice passed the full gate; all new API additive and `@ExperimentalMemoryApi` |

## M2 — 1.58-bit on a 2 GB board

| ID | Criterion | Status | Evidence |
|---|---|---|---|
| M2-A1 | BitNet-2B decodes, resident ≤ 1.3 GB, RSS flat | **planned + machinery met**, model run open | `BitNet2BPlanTest` computes the checkpoint's resident total from its geometry: **1.19 GB** with a quantized KV cache at ctx 2048, **1.29 GB** with bf16. `M2AcceptanceTest` shows the machinery (ternary weights, int8 adapter, KV ring) keeps memory flat, on CI and on the reference board. Decoding an actual BitNet checkpoint needs the model stack in SKaiNET-transformers. |
| M2-A2 | NEON `bitnet_gemv` parity 1e-5, ≥ 3× reference on Cortex-A55 | **met** (with a caveat) | parity **exact** (relative error 0 — the arithmetic is integer until the block scale). On the reference board, k=1024 n=256: reference 21.3 ms → NEON 0.091 ms. That fallback number came from a *debug* Kotlin/Native build, so treat the multiple as an upper bound; against compiler-vectorized C the intrinsics buy 1.08–1.24×. |
| M2-A3 | Page-fault rate per decode step after warm-up ≈ 0 | **met** | `M2AcceptanceTest.m2a3…` — read from `/proc/self` by `MemoryProbe`. On the reference board: **0 major faults** across 12 decode steps, RSS **14 MB before and 14 MB after**, and **0 bytes** of growth at both 4 and 48 steps. |
| M2-A4 | Ring wrap-around gives identical logits | **met** | `WindowedKvTest` — a ring that wrapped several times and a cache that never wrapped produce bit-identical output over the same window; the test first asserts the window really wrapped |
| M2-A5 | Android mmap closes #921/#922: Llama-1B Q4_K_M on a 2 GB device | open | the configuration exists (`AndroidGguf.loader()`, `staging = MAPPED`) and the fit check answers before loading, but packed weights still reach the managed heap: the packed matmul SPI takes `ByteArray`s, and a buffer-aware kernel needs the byte-order contract of #973 settled. Also needs an Android device — the reference board has no ART. |
| M2-A6 | All M1 criteria still pass | **met** | the M1 suite runs unchanged in CI and passed on the reference board alongside the M2 suite (15 tests, all green) |

## What the reference board measured

Running the M1 and M2 suites from the Kotlin/Native binary on the Cortex-A55 board — 15 tests, all
green — the ternary decode harness reported:

```
[m2] steps=12  before: rss=14 MB majflt=0 minflt=4813   after: rss=14 MB majflt=0 minflt=5037
[m2] rss growth: 4 steps → 0 bytes, 48 steps → 0 bytes
```

Zero major faults: nothing went to disk during steady-state decode. The minor faults are page-cache
touches, and the resident set is the same after forty-eight steps as after four — which is the
property M1-A1 states in allocation events and M2-A3 states in the kernel's own numbers, now
observed on the target class of device rather than inferred.

## The numbers behind M2-A1

Computed by the planner from BitNet-b1.58-2B-4T's geometry — 30 layers, 2560 hidden, 6912 FFN,
20 heads / 5 KV heads, 128 256 vocab — the same way `skainet-plan` computes them from a GGUF header:

| part | bytes |
|---|---:|
| ternary linear weights (TQ2_0, 2.0625 bits/element) | 512 MB |
| token embedding table (bf16, output head tied) | 657 MB |
| KV cache @ ctx 2048, bf16 | 150 MB |
| KV cache @ ctx 2048, TurboQuant-4 | 44 MB |
| **resident, quantized cache** | **1.19 GB** |
| **resident, bf16 cache** | **1.29 GB** |

Two things worth saying plainly:

- **The embedding table outweighs the ternary stack.** 657 MB of bf16 embeddings against 512 MB of
  1.58-bit weights: past a certain point a "2-bit model" is an embedding-table problem. The
  planner says so before anything is loaded, which is the point of M0.
- **A 2 GB device does not hold this model.** With the mobile profile's 700 MB reserve, 1.5 GB free
  leaves 800 MB, and 1.19 GB does not fit in 800 MB however it is staged. `BitNet2BPlanTest`
  asserts the refusal, and that a 4 GB device does hold it with mapped weights. M2's title is
  aspirational for *this* checkpoint; the machinery it names is in place.

## What closing M2 still needs

1. **#973** — the packed-quant byte-order contract. Until it is settled, packed weights cannot be
   handed to a kernel as a mapped view, which is what M2-A5 and the packed half of M2-A1 wait on.
2. **The decode sample in SKaiNET-transformers** — a real checkpoint, a tokenizer and a generation
   loop, which this repository deliberately does not have. M1-A2, M1-A5's tok/s and M2-A1's
   measured run belong there.
3. **An Android device** for the ART-heap criteria; the reference board answers the RAM and
   page-fault questions but not the managed-heap ones.
