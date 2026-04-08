package sk.ainet.lang.tensor.storage

/**
 * Declares placement intent for a tensor parameter or property.
 *
 * The [MemoryPlanner] reads these annotations (via reflection or codegen)
 * to decide where tensors should be allocated. This expresses *intent*,
 * not a hard guarantee — the planner may fall back if the target is
 * unavailable and [requirement] is [Requirement.PREFERRED].
 *
 * Example:
 * ```kotlin
 * @Place(device = DeviceKind.GPU, memory = MemoryDomain.DEVICE_LOCAL)
 * val projectionWeight: Tensor<FP32, Float>
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class Place(
    val device: DeviceKind = DeviceKind.AUTO,
    val memory: MemoryDomain = MemoryDomain.HOST_HEAP,
    val requirement: Requirement = Requirement.PREFERRED
)

/**
 * Marks a tensor as an immutable weight that should be file-backed
 * (memory-mapped) when possible.
 *
 * Equivalent to `@Place(device = CPU, memory = MMAP_FILE)` with
 * [Residency.PERSISTENT]. The planner treats these tensors as
 * read-only and long-lived, preferring OS-paged file access over
 * heap allocation.
 *
 * Example:
 * ```kotlin
 * @Weights
 * val embeddings: Tensor<FP32, Float>
 *
 * @Weights(memory = MemoryDomain.HOST_HEAP)  // force heap for small weights
 * val biasVector: Tensor<FP32, Float>
 * ```
 */
@Target(AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD)
@Retention(AnnotationRetention.RUNTIME)
public annotation class Weights(
    val memory: MemoryDomain = MemoryDomain.MMAP_FILE
)

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
