package sk.ainet.lang.tensor.storage

/**
 * Configures TurboQuant KV-cache compression for an attention layer.
 *
 * Applied to attention layer properties to declare KV-cache compression
 * settings. The runtime uses these annotations to configure the
 * [KvCacheStore] and [CompressedKvAttention] for each layer.
 *
 * Example:
 * ```kotlin
 * @KvCache(preset = "balanced")
 * val selfAttention: MultiHeadAttention
 *
 * @KvCache(keyBits = 8, valueBits = 4)
 * val crossAttention: MultiHeadAttention
 *
 * @KvCache(preset = "safe-lowbit", maxSeqLen = 4096)
 * val longContextAttention: MultiHeadAttention
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class KvCache(
    /**
     * Named preset: "safe-lowbit", "balanced", "experimental-max", or "none".
     * When set to a named preset, [keyBits] and [valueBits] are ignored.
     * Default "none" means no TurboQuant compression (dense FP32 cache).
     */
    val preset: String = "none",

    /**
     * Bits per element for key compression (2, 3, 4, or 8).
     * Only used when [preset] is "none" (custom config).
     */
    val keyBits: Int = 4,

    /**
     * Bits per element for value compression (2, 3, 4, or 8).
     * Only used when [preset] is "none" (custom config).
     */
    val valueBits: Int = 4,

    /**
     * Whether to use QJL residual for improved inner-product accuracy.
     * Only used when [preset] is "none" (custom config).
     */
    val useQjl: Boolean = false,

    /**
     * Maximum sequence length for the KV cache.
     * 0 means use the model's default.
     */
    val maxSeqLen: Int = 0,

    /**
     * Preferred device for KV cache storage.
     */
    val device: DeviceKind = DeviceKind.AUTO
)

/**
 * Disables TurboQuant compression for a specific layer.
 *
 * When applied alongside a model-level [KvCache] annotation, this
 * overrides the compression setting for individual layers that are
 * sensitive to quantization (e.g., early layers or cross-attention).
 *
 * Example:
 * ```kotlin
 * @KvCacheBypass
 * val firstLayerAttention: MultiHeadAttention  // stays FP32
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class KvCacheBypass
