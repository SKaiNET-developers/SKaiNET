# Vendored: NeoGPU ternary LUT kernel

Byte-identical copy of one file from the NeoGPU project, vendored under its
MIT license (see the REUSE `.license` sidecar; `LICENSES/MIT.txt` at the repo
root carries the license text). Agreed with upstream in
[anjaustin/neogpu#1](https://github.com/anjaustin/neogpu/issues/1); SKaiNET
tracking issue: [#1136](https://github.com/SKaiNET-developers/SKaiNET/issues/1136).

| | |
|---|---|
| Upstream | <https://github.com/anjaustin/neogpu> |
| File | `src/hs_ml_ternary_neon.c` |
| Vendored at commit | `0846b24ceb9f76a610b3efe2967ff4ead2ef10e6` |
| SHA-256 | `a560ffcf4d5a2e2f600d715b8193da999a92982160c625e5da48076d2257d587` |
| Local modifications | **none** |

## What it is

An f32-activation × ternary-weight ({-1, 0, +1}) matmul for AArch64 using a
256-entry × 4-float decode LUT (4 KB, L1-resident) — `vld1q_f32(lut[w[b]])` +
`vfmaq_f32` inner loop. Needs only baseline NEON (`armv8-a+simd`), no
`FEAT_DotProd`: it is the fast path for Cortex-A72 / Raspberry Pi 4-class
cores. Carries its own exact scalar fallback for non-ARM builds. The 2-bit
payload rule (4 codes per byte, low bit-pair first, code = value + 1) is
byte-identical to SKaiNET's `BITNET_B1_58` / `TernaryPacked` encoding.

Exports three symbols; SKaiNET wraps the first two via
`src/skainet_ternary_f32.c` and never calls the third
(`hs_ml_lmhead_stage1_i8` uses a thread-unsafe static scratch buffer):

- `hs_ml_ternary_f32_proj` — single-plane projection GEMV
- `hs_ml_lmhead_stage1` — fused 4-plane lm_head with FP16 row scales
- `hs_ml_lmhead_stage1_i8` — ABI-compat stub, **do not use**

## Constraints the adapter enforces / works around

- `build_lut()` uses a non-atomic init guard → the adapter warms it up under
  `pthread_once` before any concurrent use.
- pthreads: threading is NeoGPU's own (4 threads once `N >= 512` for the proj;
  the lm_head always threads). No MSVC build — the adapter compiles a portable
  scalar fallback there instead (`SKAINET_HAVE_NEOGPU_TERNARY` unset).
- Byte code 3 decodes to +2.0 (matches `TernaryCodec.decodeBitNet`); loaders
  must reject code 3 at import, the kernel never validates.
- Must be compiled at `-march=armv8-a` on non-Apple AArch64 — the library
  default `-march=armv8.2-a+fp16+dotprod` would defeat its purpose
  (dotprod-less targets).

## Re-vendoring

Copy the file byte-identical from upstream, update the commit + SHA-256 here,
keep "Local modifications: none" true (fixes belong in the adapter or
upstream), and re-run the ternary goldens in
`src/jvmTest/kotlin/sk/ainet/exec/kernel/NativeTernaryF32GemvKernelTest.kt`.
