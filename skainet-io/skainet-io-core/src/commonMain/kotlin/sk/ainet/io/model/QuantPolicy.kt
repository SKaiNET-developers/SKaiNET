package sk.ainet.io.model

/**
 * Controls how quantized tensors are handled during weight loading.
 *
 * Shared across all weight loaders (LLaMA, Gemma, etc.).
 */
@Deprecated(
    "One of the three axes WeightForm replaces (#1109): this is the encoding axis. No ReplaceWith, " +
        "because EncodingRequest is not a drop-in — NATIVE_OPTIMIZED becomes KeepAsStored, " +
        "DEQUANTIZE_TO_FP32 becomes DequantizeTo(FP32), and RAW_BYTES has no counterpart because " +
        "no loader ever supported it. A wrong ReplaceWith would be worse than none.",
    level = DeprecationLevel.WARNING,
)
public enum class QuantPolicy {
    /** Keep quantized payloads as raw bytes (Int8 tensor) with quantized shape. */
    RAW_BYTES,

    /** Dequantize to FP32 on load. */
    DEQUANTIZE_TO_FP32,

    /**
     * Mixed mode: dequantize F32/F16/BF16 tensors to FP32, but keep quantized
     * weight tensors (Q4_0, Q8_0, etc.) as raw bytes for native kernel consumption.
     *
     * This allows loading with dtype=FP32 while preserving quantized weights
     * for platform-specific optimized kernels (e.g. MemorySegment-backed SIMD).
     */
    NATIVE_OPTIMIZED,
}
