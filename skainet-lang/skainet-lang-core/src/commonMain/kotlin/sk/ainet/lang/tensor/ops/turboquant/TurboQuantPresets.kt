package sk.ainet.lang.tensor.ops.turboquant

import sk.ainet.lang.tensor.storage.KvCacheConfig
import sk.ainet.lang.tensor.storage.Placement
import sk.ainet.lang.tensor.storage.Residency
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Named preset configurations for TurboQuant KV-cache compression.
 *
 * Presets reflect the practical observation that key precision is often
 * more quality-sensitive than value precision.
 *
 * Available presets:
 * - **safe-lowbit**: Q8_0 keys + TurboQuant-4 values (conservative)
 * - **balanced**: TurboQuant-4 keys + TurboQuant-4 values
 * - **experimental-max**: TurboQuant-3 keys + TurboQuant-3 values (aggressive)
 */
public object TurboQuantPresets {

    /**
     * Safe low-bit preset: Q8_0 for keys, TurboQuant-4 for values.
     *
     * Keys stay at 8-bit for quality preservation; values are compressed
     * to 4-bit TurboQuant. Good for production use where key accuracy
     * matters more than value accuracy.
     */
    public fun safeLowbit(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ): TurboQuantPreset = TurboQuantPreset(
        name = "safe-lowbit",
        cacheConfig = KvCacheConfig(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            keyEncoding = TensorEncoding.Q8_0,
            valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4),
            placement = Placement.CPU_HEAP.copy(residency = Residency.PERSISTENT)
        ),
        keyQuantConfig = null, // Q8_0 uses standard quantization, not TurboQuant
        valueQuantConfig = TurboQuantConfig.polarOnly(bits = 4)
    )

    /**
     * Balanced preset: TurboQuant-4 for both keys and values.
     *
     * Symmetric 4-bit compression for both K and V. Good balance
     * between compression ratio and quality.
     */
    public fun balanced(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ): TurboQuantPreset = TurboQuantPreset(
        name = "balanced",
        cacheConfig = KvCacheConfig(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4),
            valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 4),
            placement = Placement.CPU_HEAP.copy(residency = Residency.PERSISTENT)
        ),
        keyQuantConfig = TurboQuantConfig.polarOnly(bits = 4),
        valueQuantConfig = TurboQuantConfig.polarOnly(bits = 4)
    )

    /**
     * Experimental maximum compression: TurboQuant-3 for both K and V.
     *
     * Aggressive 3-bit compression. Use with caution — may degrade quality
     * for some models. Best suited for long-context scenarios where memory
     * is the primary constraint.
     */
    public fun experimentalMax(
        numLayers: Int,
        numHeads: Int,
        headDim: Int,
        maxSeqLen: Int
    ): TurboQuantPreset = TurboQuantPreset(
        name = "experimental-max",
        cacheConfig = KvCacheConfig(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 3),
            valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = 3),
            placement = Placement.CPU_HEAP.copy(residency = Residency.PERSISTENT)
        ),
        keyQuantConfig = TurboQuantConfig.polarOnly(bits = 3),
        valueQuantConfig = TurboQuantConfig.polarOnly(bits = 3)
    )

    /**
     * List all available preset names.
     */
    public val availablePresets: List<String> = listOf("safe-lowbit", "balanced", "experimental-max")
}

/**
 * A named TurboQuant preset with all configuration needed to create a cache.
 */
public data class TurboQuantPreset(
    val name: String,
    val cacheConfig: KvCacheConfig,
    /** TurboQuant config for keys, or null if keys use non-TurboQuant encoding. */
    val keyQuantConfig: TurboQuantConfig?,
    /** TurboQuant config for values, or null if values use non-TurboQuant encoding. */
    val valueQuantConfig: TurboQuantConfig?
)
