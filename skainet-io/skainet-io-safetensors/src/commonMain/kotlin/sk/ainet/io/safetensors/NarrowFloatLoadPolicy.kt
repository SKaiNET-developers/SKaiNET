package sk.ainet.io.safetensors

/**
 * Controls how a loader handles **16-bit float** source tensors (`FLOAT16` / `BFLOAT16`).
 *
 * The choice is whether to **widen to FP32 at load time** — doubling the resident footprint of
 * every narrow tensor — or to **keep the on-disk bytes** and let the matmul dispatch route to a
 * narrow-float kernel that widens per-multiply instead.
 *
 * Both paths are numerically equivalent when they reach the same kernel: the narrow → FP32
 * conversion is exact in either direction (FP32 is a strict superset of both formats), applied
 * either once at load ([DEQUANT_TO_FP32]) or per-multiply ([KEEP_NATIVE]).
 *
 * This supersedes the BF16-only `Bf16LoadPolicy`, which remains as a typealias.
 */
public enum class NarrowFloatLoadPolicy {
    /**
     * Default. Widen every narrow-float tensor to FP32 at load time and wrap it as
     * `FloatArrayTensorData`, exactly as a `FLOAT32` source tensor would be.
     *
     * Memory: 2× the on-disk size for each narrow tensor. Runtime: no extra dispatch — every
     * matmul receives FP32 operands.
     */
    DEQUANT_TO_FP32,

    /**
     * Keep narrow-float tensors in their on-disk packed 2-bytes-per-element layout. The loader
     * emits a `NarrowFloatDenseTensorData` (`Bf16DenseTensorData` / `Fp16DenseTensorData`) rather
     * than widening. The tensor still presents `Float` on read — `get` decodes — but its
     * `tensor.data` is recognisable as `NarrowFloatTensorData`, which is what lets a matmul
     * dispatch pick the narrow-float SPI kernel.
     *
     * Memory: identical to the on-disk size — no doubling. Runtime: the narrow kernel is used when
     * one is registered; otherwise consumers fall back to per-element decode.
     *
     * **Caveat**: any non-matmul op touching the tensor pays a per-element decode via `get`. Worth
     * it when the hot path is matmul-dominated (the typical transformer case), not otherwise.
     */
    KEEP_NATIVE,
}
