package sk.ainet.models.gemma

import sk.ainet.models.gemma.Gemma3nModelMetadata
import sk.ainet.models.gemma.LayerType

/**
 * Configuration for Gemma 3n model inference.
 *
 * This class provides configuration parameters for the Gemma 3n E2B model,
 * including its unique hybrid attention pattern and variable FFN dimensions.
 *
 * Key architecture differences from standard Llama:
 * - GELU activation instead of SiLU
 * - Hybrid attention: 4 local sliding-window + 1 global full attention (repeating)
 * - Variable FFN sizes per layer (MatFormer architecture)
 * - Per-layer embeddings (optional)
 * - KV cache sharing for the last N layers
 * - Dual RoPE frequencies: 10k for local, 1M for global attention
 *
 * @property hiddenSize Main hidden dimension (d_model)
 * @property perLayerHiddenSize Per-layer embedding dimension (optional)
 * @property numLayers Number of transformer layers
 * @property numAttentionHeads Number of query attention heads
 * @property numKvHeads Number of key-value heads (for GQA)
 * @property headDim Dimension per attention head
 * @property intermediateSizes FFN intermediate dimension per layer (variable)
 * @property slidingWindow Local attention window size
 * @property layerPattern Pattern of attention types (e.g., ["sliding", "sliding", "sliding", "sliding", "full"])
 * @property ropeBaseLocal RoPE base frequency for local/sliding attention
 * @property ropeBaseGlobal RoPE base frequency for global attention
 * @property kvSharedLayers Number of last layers that share KV cache
 */
public data class Gemma3nConfig(
    val hiddenSize: Int = 2048,
    val perLayerHiddenSize: Int = 256,
    val numLayers: Int = 35,
    val numAttentionHeads: Int = 8,
    val numKvHeads: Int = 2,
    val headDim: Int = 256,
    val intermediateSizes: List<Int> = List(35) { 8192 },
    val slidingWindow: Int = 512,
    val layerPattern: List<String> = DEFAULT_LAYER_PATTERN,
    val ropeBaseLocal: Float = 10000f,
    val ropeBaseGlobal: Float = 1000000f,
    val kvSharedLayers: Int = 15
) {
    /** Total query dimension (numAttentionHeads * headDim) */
    public val queryDim: Int get() = numAttentionHeads * headDim

    /** Total key-value dimension (numKvHeads * headDim) */
    public val kvDim: Int get() = numKvHeads * headDim

    /** Number of query heads per KV head for GQA */
    public val numHeadsPerKv: Int get() = numAttentionHeads / numKvHeads

    /**
     * Returns the layer type at the given layer index.
     */
    public fun getLayerType(layerIdx: Int): LayerType {
        val patternIdx = layerIdx % layerPattern.size
        return when (layerPattern[patternIdx]) {
            "full", "global" -> LayerType.GLOBAL
            else -> LayerType.SLIDING
        }
    }

    /**
     * Returns whether the layer uses sliding window attention.
     */
    public fun isLocalLayer(layerIdx: Int): Boolean = getLayerType(layerIdx) == LayerType.SLIDING

    /**
     * Returns whether the layer uses global full attention.
     */
    public fun isGlobalLayer(layerIdx: Int): Boolean = getLayerType(layerIdx) == LayerType.GLOBAL

    /**
     * Returns the RoPE base frequency for the given layer.
     */
    public fun getRopeBase(layerIdx: Int): Float {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> ropeBaseGlobal
            LayerType.SLIDING -> ropeBaseLocal
        }
    }

    /**
     * Returns the FFN intermediate dimension for the given layer.
     */
    public fun getIntermediateSize(layerIdx: Int): Int {
        return intermediateSizes.getOrElse(layerIdx) {
            intermediateSizes.lastOrNull() ?: (hiddenSize * 4)
        }
    }

    /**
     * Returns whether the given layer shares its KV cache.
     */
    public fun isKvShared(layerIdx: Int): Boolean {
        return layerIdx >= (numLayers - kvSharedLayers)
    }

    /**
     * Returns the cache layer index for the given layer.
     * Shared layers map to a single cache slot.
     */
    public fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (isKvShared(layerIdx)) {
            numLayers - kvSharedLayers
        } else {
            layerIdx
        }
    }

    /**
     * Returns the effective number of cache layers (accounting for sharing).
     */
    public val effectiveCacheLayers: Int
        get() = (numLayers - kvSharedLayers) + 1

    public companion object {
        /** Default layer pattern: 4 sliding + 1 global, repeating */
        public val DEFAULT_LAYER_PATTERN: List<String> = listOf("sliding", "sliding", "sliding", "sliding", "full")

        /**
         * Create config from loaded model metadata.
         */
        public fun fromMetadata(metadata: Gemma3nModelMetadata): Gemma3nConfig {
            return Gemma3nConfig(
                hiddenSize = metadata.embeddingLength,
                perLayerHiddenSize = metadata.perLayerEmbeddingLength,
                numLayers = metadata.blockCount,
                numAttentionHeads = metadata.headCount,
                numKvHeads = metadata.kvHeadCount,
                headDim = metadata.headDim,
                intermediateSizes = metadata.feedForwardLengths,
                slidingWindow = metadata.slidingWindow,
                layerPattern = metadata.layerPattern,
                ropeBaseLocal = metadata.ropeBaseLocal,
                ropeBaseGlobal = metadata.ropeBaseGlobal,
                kvSharedLayers = metadata.kvSharedLayers
            )
        }

        /**
         * Default configuration for Gemma 3n E2B model.
         */
        public val E2B_DEFAULT: Gemma3nConfig = Gemma3nConfig(
            hiddenSize = 2048,
            perLayerHiddenSize = 256,
            numLayers = 35,
            numAttentionHeads = 8,
            numKvHeads = 2,
            headDim = 256,
            intermediateSizes = List(35) { 8192 },
            slidingWindow = 512,
            layerPattern = DEFAULT_LAYER_PATTERN,
            ropeBaseLocal = 10000f,
            ropeBaseGlobal = 1000000f,
            kvSharedLayers = 15
        )
    }
}
