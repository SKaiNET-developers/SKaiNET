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

## Status

`Layout.blockOrder`, the `LayoutClass` split, `prepack` and this document land with
[#1094](https://github.com/SKaiNET-developers/SKaiNET/issues/1094). The remaining work is tracked
as sub-issues of #973: bridging the packed SPI kernels through the ordered key
([#1095](https://github.com/SKaiNET-developers/SKaiNET/issues/1095)), the weight-transposing matmul
primitive (#1096), engine-owned prepacking and cross-repo contract fixtures
([#1097](https://github.com/SKaiNET-developers/SKaiNET/issues/1097)), and normalizing weight
orientation at the load boundary ([#1098](https://github.com/SKaiNET-developers/SKaiNET/issues/1098)).
