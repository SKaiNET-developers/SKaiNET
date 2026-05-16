package sk.ainet.io.safetensors

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
public enum class Bf16LoadPolicy {
    /**
     * Default. Dequantise every BFLOAT16 tensor to FP32 at load time
     * via the existing `dequantBF16` helper, then wrap as a
     * `FloatArrayTensorData` (same as `FLOAT32` source tensors).
     *
     * Memory cost: 2× the on-disk size for each BF16 tensor.
     * Runtime: zero extra dispatch — every matmul gets FP32 operands.
     */
    DEQUANT_TO_FP32,

    /**
     * Keep BFLOAT16 tensors in their on-disk packed-2-bytes-per-element
     * layout. The loader emits a `Bf16DenseTensorData` (in
     * `skainet-lang-core`) instead of dequanting; the tensor still
     * advertises FP32 dtype to consumers (the underlying `get` decodes
     * on read), but its `tensor.data` is recognisable as
     * `Bf16TensorData` so a matmul dispatch can route to the SIMD
     * `Bf16MatmulKernel` SPI.
     *
     * Memory cost: identical to the on-disk size — no doubling.
     * Runtime: matmul dispatch picks up the BF16 SPI kernel when one
     * is registered; falls back to per-multiply dequant otherwise.
     *
     * **Caveat**: any non-matmul op that touches the BF16 tensor pays
     * a per-element decode cost via `get`. Don't flip this unless the
     * model's hot path is dominated by matmuls (the typical
     * transformer case).
     */
    KEEP_NATIVE,
}
