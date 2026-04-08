package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.ops.turboquant.TurboQuantConfig
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantPresets

/**
 * Resolves [KvCache] annotations to [KvCacheStore] instances.
 *
 * Used by skainet-transformers to create KV caches declaratively.
 * When a model layer is annotated with `@KvCache(preset = "balanced")`,
 * this resolver creates the appropriate compressed or dense cache.
 *
 * Example:
 * ```kotlin
 * // In skainet-transformers attention layer:
 * @KvCache(preset = "balanced")
 * class SelfAttention(val numHeads: Int, val headDim: Int, ...) {
 *     val cache = KvCacheAnnotationResolver.resolve(
 *         annotation = this::class.annotations.filterIsInstance<KvCache>().first(),
 *         numLayers = modelConfig.numLayers,
 *         numHeads = numHeads,
 *         headDim = headDim,
 *         maxSeqLen = modelConfig.maxSeqLen
 *     )
 * }
 * ```
 */
public object KvCacheAnnotationResolver {

    /**
     * Resolve a [KvCache] annotation to a [KvCacheStore].
     *
     * @param annotation The @KvCache annotation values
     * @param numLayers  Number of transformer layers
     * @param numHeads   Number of KV heads per layer
     * @param headDim    Dimension per head
     * @param maxSeqLen  Maximum sequence length (overridden by annotation if > 0)
     */
    public fun resolve(
        annotation: KvCache,
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ): KvCacheStore {
        val effectiveMaxSeqLen = if (annotation.maxSeqLen > 0) annotation.maxSeqLen else maxSeqLen

        return when (annotation.preset) {
            "none" -> {
                // Custom config from annotation parameters
                KvCacheStore.turboQuant(
                    numLayers = numLayers,
                    numHeads = numHeads,
                    headDim = headDim,
                    maxSeqLen = effectiveMaxSeqLen,
                    keyBits = annotation.keyBits,
                    valueBits = annotation.valueBits,
                    useQjl = annotation.useQjl
                )
            }
            "dense" -> {
                KvCacheStore.dense(numLayers, numHeads, headDim, effectiveMaxSeqLen)
            }
            else -> {
                // Named preset
                KvCacheStore.turboQuant(
                    preset = annotation.preset,
                    numLayers = numLayers,
                    numHeads = numHeads,
                    headDim = headDim,
                    maxSeqLen = effectiveMaxSeqLen
                )
            }
        }
    }

    /**
     * Resolve a preset name string to a [KvCacheStore].
     *
     * Convenience for when you have the preset name but not the full annotation.
     *
     * @param preset     "dense", "safe-lowbit", "balanced", or "experimental-max"
     * @param numLayers  Number of transformer layers
     * @param numHeads   Number of KV heads per layer
     * @param headDim    Dimension per head
     * @param maxSeqLen  Maximum sequence length
     */
    public fun resolve(
        preset: String,
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ): KvCacheStore = when (preset) {
        "dense", "none" -> KvCacheStore.dense(numLayers, numHeads, headDim, maxSeqLen)
        else -> KvCacheStore.turboQuant(preset, numLayers, numHeads, headDim, maxSeqLen)
    }
}
