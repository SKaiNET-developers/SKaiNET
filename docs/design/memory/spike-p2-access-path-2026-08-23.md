# Phase-2 spike — TensorView / Storage access-path cost on the JVM — 2026-08-23

SKEEP-003 decision #6 / issue #1016. Before milestone M1 introduces `Storage`, `Scope` and `TensorView`, measure what a view layer between a kernel and its bytes costs, with a throw-away model of the proposed types: `TensorViewSpikeBench` and `FlatRssSpike` in `skainet-backends/benchmarks/jvm-cpu-jmh` (package `sk.ainet.bench.spike`; run with `-PjmhIncludes='spike.*'` and `runFlatRssSpike`). Budget (decision #6): elementwise ≤ 3 %, matmul within noise; above that, the access path is redesigned — unwrap to the raw array / segment once per call — before Phase 2 proper.

- Commit: `develop` @ 3c7c7d19 + the spike benchmarks · HotSpot JDK 25 (`--enable-preview --add-modules jdk.incubator.vector`) · JMH fork 1 / 3 warmup / 5 iterations · i7-9750H, machine idle.
- Model: `SpikeStorage` sealed (`Heap(FloatArray)` | `OffHeap(MemorySegment)`), `SpikeView(storage, offset, length)` with `get`/`set` dispatching on the storage kind, and the two fast paths a kernel takes once per call: `asHeapArray()` and `segment()`.

## Elementwise `c = a + b`, 1 M floats (avgt, µs/op — lower is better)

| Variant | µs/op | vs raw |
|---|---:|---:|
| raw `FloatArray`, zero-based loop (`add_raw`) | 286 ± 16 | +0 % |
| raw `FloatArray`, loop-invariant offsets — **control, no view** (`add_rawOffset`) | 539 ± 24 | +88 % |
| view over heap, `asHeapArray()` once, offset loop (`add_viewHeapUnwrap`) | 567 ± 182 | +98 % |
| view over heap, `get`/`set` per element (`add_viewHeapGet`) | 516 ± 15 | +81 % |
| view over `MemorySegment`, `segment()` once, `getAtIndex` per element (`add_viewOffHeapUnwrap`) | 495 ± 19 | +73 % |
| view over `MemorySegment`, `get`/`set` per element (`add_viewOffHeapGet`) | 1579 ± 3 | +452 % |

## GEMV 256 × 1024 (avgt, µs/op)

| Variant | µs/op | vs raw |
|---|---:|---:|
| raw arrays (`gemv_raw`) | 249 ± 1 | +0 % |
| view over heap, unwrap once (`gemv_viewHeapUnwrap`) | 253 ± 1 | +1 % |
| view over heap, `get` per element (`gemv_viewHeapGet`) | 249 ± 1 | +0 % |
| view over `MemorySegment`, unwrap once (`gemv_viewOffHeapUnwrap`) | 257 ± 2 | +3 % |
| view over `MemorySegment`, `get` per element (`gemv_viewOffHeapGet`) | 259 ± 2 | +4 % |

## Flat-RSS decode loop — `Forward` slab vs heap arrays per step (3 000 steps, `-Xmx256m`)

```
FlatRssSpike: 3000 steps, per-step activations = 1 MiB, pid=3758280
--- Forward slab (Arena.ofShared bump + reset per step) ---
  step     1  RSS     62 MiB  heapUsed    9 MiB  slab allocations so far 193 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step    50  RSS     65 MiB  heapUsed   10 MiB  slab allocations so far 9650 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step   100  RSS     66 MiB  heapUsed   10 MiB  slab allocations so far 19300 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step   500  RSS     69 MiB  heapUsed   13 MiB  slab allocations so far 96500 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step  1000  RSS     73 MiB  heapUsed   17 MiB  slab allocations so far 193000 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step  1500  RSS     77 MiB  heapUsed   21 MiB  slab allocations so far 289500 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step  2000  RSS     78 MiB  heapUsed   21 MiB  slab allocations so far 386000 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  step  3000  RSS     86 MiB  heapUsed   29 MiB  slab allocations so far 579000 (slab bytes allocated per step after warm-up: 0; each allocate() still creates one MemorySegment view object)
  sink=8.64284E8
--- Heap arrays per step (today: GC-backed) ---
  step     1  RSS     86 MiB  heapUsed   31 MiB  heap allocations so far 193
  step    50  RSS    141 MiB  heapUsed   82 MiB  heap allocations so far 9650
  step   100  RSS    174 MiB  heapUsed   71 MiB  heap allocations so far 19300
  step   500  RSS    213 MiB  heapUsed  123 MiB  heap allocations so far 96500
  step  1000  RSS    213 MiB  heapUsed   74 MiB  heap allocations so far 193000
  step  1500  RSS    214 MiB  heapUsed   15 MiB  heap allocations so far 289500
  step  2000  RSS    214 MiB  heapUsed  111 MiB  heap allocations so far 386000
  step  3000  RSS    214 MiB  heapUsed  149 MiB  heap allocations so far 579000
  sink=8.64284E8
```

## What the numbers say

1. **The view indirection itself is within budget when a kernel unwraps once per call.** GEMV: heap unwrap +1 %, heap `get` 0 %, off-heap +3–4 % (the dot-product reduction is not auto-vectorized, so per-element overhead is hidden). Elementwise: `add_viewHeapUnwrap` (566 ± 182) ≈ `add_rawOffset` (539 ± 24) — the view adds nothing measurable on top of its control.
2. **What *is* expensive is offset-based indexing, not the view:** `add_rawOffset` — raw arrays, no view, just loop-invariant offsets — is 1.9× slower than the zero-based loop. C2's auto-vectorizer (SuperWord) handles the zero-based `c[i] = a[i] + b[i]` and does not handle the three-offset form here. Consequence for M1: JVM elementwise kernels operating on views must not depend on C2 auto-vectorization; SKaiNET's vector kernels (`JvmVectorKernels`, explicit `FloatVector.fromArray(species, array, offset)`) take offsets explicitly and are unaffected — the ≤ 3 % elementwise budget is to be *measured on those kernels* when the façades land (S1.4 gate re-runs `ElementwiseAdd1MBench`), and scalar fallbacks should special-case `offset == 0`.
3. **Per-element `get()` through a view is the slow path, by design.** +80 % over heap, 5.5× over `MemorySegment` on a vectorizable loop. Rule 4 (decoding `get()`) stays the reference path for correctness; production kernels receive a `TensorView` and call `asHeapArray()` / `segment()` once.
4. **Off-heap activations on the JVM need Vector-API-over-segment kernels.** Scalar `getAtIndex` loops over `MemorySegment` are 1.7× (unwrapped) to 5.5× (through the view) slower than arrays. Weights are already consumed as segments by the Panama quantized kernels (`ByteVector.fromMemorySegment`), so mapped/off-heap **weights** are fine; for **activations**, `Scope.Forward`'s JVM binding should default to a **heap slab** (recycled `FloatArray`s / bump-allocated arrays) and make off-heap opt-in until the elementwise kernels have segment variants. Input for S1.1b / S1.2.
5. **Flat RSS:** with a recycled bump slab the process stays at 62–86 MiB over 3 000 steps with zero slab bytes allocated after warm-up; with fresh heap arrays per step it climbs to 214 MiB and plateaus there (GC heap expanded to its ceiling). The slab run's residual growth (heapUsed 9 → 29 MiB) is the one `MemorySegment` view object `allocate()` creates per activation (193 per step) — the real `TensorView` will be a small object too: cheap, not free; pool or reuse views on the hot loop if minor GCs show up in the M1 trace.

## Verdict (decision #6)

**Go for Phase 2**, with the access-path rule written into the M1 slices: kernels take `TensorView`s and unwrap once (`asHeapArray()` / `segment()`); `get()` decodes and is the reference path only; JVM `Forward` scope allocates heap slabs by default; vector kernels keep explicit offsets. The elementwise ≤ 3 % budget is re-checked on the real vector kernels at S1.4 (façades) and S1.7 (dispatch) against `docs/design/memory/baseline-2026-08-22.md`.

## Open: Android half

The same measurement on a Cortex-A55-class 2 GB device (direct `ByteBuffer` path) is pending the reference-device choice (PRD §9 open item). The JVM result already sets the rule (unwrap once, heap activations by default); the Android run decides whether direct `ByteBuffer` activations are viable there and is recorded on #1016 when the device is available.
