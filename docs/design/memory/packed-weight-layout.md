# Packed weight layout — the normative contract

Every packed (block-quantized) weight in SKaiNET is stored in **one of exactly two block orders**,
and which one it is in is a property of the value, not of the type it happens to have or of the
module that produced it. This document is that contract; kdocs link here instead of restating it.

Written for [#973](https://github.com/SKaiNET-developers/SKaiNET/issues/973), whose census found
seven mutually contradicting statements of it inside this repository alone — two of which had
already produced wrong numbers in production ([#968](https://github.com/SKaiNET-developers/SKaiNET/issues/968),
[#971](https://github.com/SKaiNET-developers/SKaiNET/issues/971)).

## The two orders

A 2-D weight is logically `[out, in]`. Its blocks tile the **input** dimension, so a row of
`in` elements is `blocksPerRow = in / blockSize` blocks, and the whole weight is an
`out × blocksPerRow` grid of blocks. Flattening that grid is where the two orders come from:

| order | flat block index | who produces it | who reads it |
|---|---|---|---|
| `BlockOrder.ROW_MAJOR` (canonical) | `o * blocksPerRow + b` | GGUF files, `Q*Quantizer`, `TernaryCodec` | `toFloatArray()`, `get()`, the reference matmul, `bitnet_gemv` |
| `BlockOrder.INPUT_BLOCK_MAJOR` (kernel feed order) | `b * out + o` | `TensorView.prepack(INPUT_BLOCK_MAJOR)` | the packed matmul kernels — scalar, Panama, native C, JNI |

**They coincide only when `blocksPerRow == 1`.** For any weight wider than one block — that is,
virtually every real weight — reading one order as the other produces a block-permuted matrix:
finite, plausible numbers that are simply wrong. Nothing crashes. That is the entire reason this
document exists.

## Where the order lives

`Layout.blockOrder`. A `TensorView` over packed bytes carries it, and the order is expressed **in
the strides**, not in a branch: input-block-major is `strides = [1, out]` over the block grid
instead of `[blocksPerRow, 1]`. Everything else — `narrow`, `transpose`, `get`, `toFloatArray` —
therefore works unchanged on a view in either order, and a prepacked view still describes the same
matrix.

`OperandKey`'s `LayoutClass` splits the same way: `BLOCKED_ROW_MAJOR` and `BLOCKED_INPUT_MAJOR`.
A kernel **declares** which order it reads, in its `KernelKey`. That is what makes the dispatcher
able to insert a relayout instead of the caller having to know.

## Converting between them

`TensorView.prepack(order, scope, sink)`:

- returns `this` when the order already matches — the only free case;
- otherwise copies the blocks into `scope` and emits `TraceEvent.AdapterInserted` with the byte
  count, so the conversion shows up in the trace with its price;
- the result carries the new order on its layout.

It is a *conversion*, and it is named as one. It is not `transpose`: a true transpose of a
block-quantized weight would need runs of quantized values along the other axis, i.e.
requantization. What the engine historically called a "packed transpose" was this conversion
wearing transpose's name and a swapped shape label that lied about the data
(see #973, "the deeper semantic problem"); replacing that with a weight-transposing matmul
primitive is [#1096](https://github.com/SKaiNET-developers/SKaiNET/issues/1096).

## For a downstream repository

Two things are published with the engine so a converter never has to reimplement this:

- **`PackedWeights`** — `prepackForMatmul(view)` / `toCanonical(view)` for views, and
  `toKernelOrder(bytes, rows, blocksPerRow, bytesPerBlock)` / `toCanonicalOrder(...)` for a
  converter that holds bytes. This is the *only* sanctioned implementation of the permutation. A
  private copy is what #973 exists to stop: the census found one that had drifted from the shared
  packer it was copied from, and layout knowledge living in the wrong repository.
- **`PackedLayoutFixtures`** — canonical and kernel-order fixtures per format, in `main` rather than
  a test source set, so the artifact a downstream repository already depends on carries them.
  `disagreement(bytes, encoding, kernelOrder)` returns `null` when the bytes agree, or names the
  first block that is in the wrong place.

A downstream test asserting `disagreement(myConverterOutput, Q4_K, kernelOrder = true) == null` is
running against the same bytes the engine's own tests run against. That is what makes a layout
change fail somewhere rather than ship: previously each repository's suite proved only its own
convention, and neither crossed the boundary — which is how a byte-layout change shipped as a
green-CI hotfix.

Every fixture is three blocks wide on purpose. At one block per row the two orders coincide, and a
test built that way passes whichever convention the code holds.

## Orientation at the load boundary

A packed weight is logically `[out, in]`, and its blocks tile **`in`**. GGUF writes dimensions in
`ne` order — fastest-varying first — so the same weight arrives labelled `[in, out]`, while its
*bytes* are already `[out, in]` row-major. Only the label is wrong, and the label is what the
relayout reads: driven by `[in, out]` it permutes the wrong grid, or refuses because `out` is not a
multiple of the block size. Both failures are in #973's census.

- `StreamingGgufParametersLoader(weightOrientation = WeightOrientation.OUT_IN)` fixes the label at
  the boundary, reversing 2-D weights only. Nothing about the bytes changes. It defaults to
  `AS_STORED` — today's behaviour — because reversing shapes changes what every consumer sees; new
  code should ask for `OUT_IN`.
- `PackedWeights.requireOutIn(rows, inputDim, encoding)` refuses a weight that looks transposed
  instead of computing a wrong permutation from it, and names the fix. `prepackForMatmul` runs it.
  The check is a heuristic and says so: it fires when the *first* dimension is block-aligned and the
  second is not, which is exactly the shape `ne` order produces, and stays quiet when both are
  aligned and it cannot tell.

Note the two GGUF readers in this repository disagree about this today: the legacy `GGUFReader`
reverses dimensions, the streaming one does not. `WeightOrientation` is how a caller states which it
wants rather than discovering it.

## Rules

1. **A file's bytes are `ROW_MAJOR`.** Anything loaded from GGUF, produced by a quantizer, or
   written by `TernaryCodec` is canonical. A loader never silently relayouts.
2. **A kernel declares its order in its key.** No kernel may assume; no caller may guess.
3. **A conversion is visible.** It allocates in a scope and emits an adapter event. A relayout that
   does not appear in the trace is a bug.
4. **`get()` and `toFloatArray()` always mean the same thing** in either order — they read through
   the layout. A view whose decoded content depends on which module produced it is a bug.
5. **`packedData` byte semantics are public API.** Changing the order a type holds is a
   minor/major change, never a patch — this is what let a byte-layout change ship as a green-CI
   hotfix once already.

## Prepack once, at load

The relayout is O(bytes). A weight prepacked at load hits the packed kernel's key directly and the
dispatcher copies nothing per call; a canonical weight handed straight to the dispatcher gets the
decoding reference kernel — correct, and slower.

`KernelDispatch.matmul(..., prepackWeights = true)` will relayout for you, and it is **off by
default** because doing it inside a decode step copies the whole weight per token. That is the same
per-forward copy #973 objects to in `ops.transpose`, merely moved; wiring it on by default broke
M1-A3 in exactly that way during #1095, which is how the default was chosen.

## Status

`Layout.blockOrder`, the `LayoutClass` split, `prepack` and this document landed with
[#1094](https://github.com/SKaiNET-developers/SKaiNET/issues/1094); the packed SPI kernels reach the
registry through the ordered key since
[#1095](https://github.com/SKaiNET-developers/SKaiNET/issues/1095); `PackedWeights` and
`PackedLayoutFixtures` since [#1097](https://github.com/SKaiNET-developers/SKaiNET/issues/1097).

Weight orientation at the load boundary is opt-in since
[#1098](https://github.com/SKaiNET-developers/SKaiNET/issues/1098), with a guard that refuses a
wrongly-labelled weight rather than mis-permuting it.

Still open under #973: the weight-transposing matmul primitive that removes the packed
`ops.transpose` entirely ([#1096](https://github.com/SKaiNET-developers/SKaiNET/issues/1096)), and
making `OUT_IN` the default once downstream consumers have moved.
