package sk.ainet.io.safetensors

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32

/**
 * Controls how the SafeTensors loader handles `BFLOAT16` (BF16) tensors.
 *
 * BF16 weights are common in modern transformer checkpoints (Gemma-3n,
 * many Llama-derivatives shipped from HuggingFace). The decision is
 * whether to **dequantise to FP32 at load time** (status quo — every
 * BF16 weight doubles its memory footprint and the dequant pass runs
 * once per checkpoint load) or to **keep the BF16 bytes native** and
 * let the matmul dispatch in `DefaultCpuOps` route to a vectorised
 * BF16 kernel.
 *
 * The two paths produce numerically equivalent results when both reach
 * the same matmul kernel — the BF16 → FP32 conversion is the bit-shift
 * identity `float_bits = (bf16 & 0xFFFF) shl 16`, applied either at
 * load (DEQUANT) or per-multiply (KEEP_NATIVE).
 *
 * Existing consumers should keep the default. Flip to [KEEP_NATIVE]
 * **only** after confirming the runtime dispatch has BF16 support — at
 * the time of writing that's SKaiNET-transformers builds against
 * SKaiNET develop with the BF16 dispatch wired into
 * `DefaultCpuOpsJvm.chooseQuantizedMatmul` (Phase 3 follow-up,
 * separately tracked).
 */
public typealias Bf16LoadPolicy = NarrowFloatLoadPolicy

/**
 * Maps this narrow-float policy onto the generalised [DTypePolicy] sealed type.
 * [NarrowFloatLoadPolicy.DEQUANT_TO_FP32] becomes `Require(FP32)` (the loader must hand consumers
 * an FP32 tensor); [NarrowFloatLoadPolicy.KEEP_NATIVE] becomes `Require(BF16)`.
 *
 * Kept BF16-specific for source compatibility with existing call sites. For FP16, build the
 * policy directly (`DTypePolicy.Require(FP16)`) — see `SafeTensorsParametersLoader.mapPolicyToFp16`.
 */
public fun Bf16LoadPolicy.toDTypePolicy(): DTypePolicy = when (this) {
    NarrowFloatLoadPolicy.DEQUANT_TO_FP32 -> DTypePolicy.Require(FP32)
    NarrowFloatLoadPolicy.KEEP_NATIVE -> DTypePolicy.Require(BF16)
}
