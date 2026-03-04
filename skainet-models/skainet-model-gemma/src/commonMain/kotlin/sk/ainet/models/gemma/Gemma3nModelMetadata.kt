package sk.ainet.models.gemma

/**
 * Metadata for Gemma 3n models extracted from GGUF files.
 *
 * Gemma 3n has several unique characteristics:
 * - Variable intermediate (FFN) sizes per layer (MatFormer architecture)
 * - Hybrid attention pattern (4 local sliding-window + 1 global)
 * - Per-layer embeddings (optional)
 * - KV cache sharing for the last N layers
 */
public data class Gemma3nModelMetadata(
    val architecture: String,
    val embeddingLength: Int,
    val perLayerEmbeddingLength: Int,
    val contextLength: Int,
    val blockCount: Int,
    val headCount: Int,
    val kvHeadCount: Int,
    val feedForwardLengths: List<Int>,
    val headDim: Int,
    val vocabSize: Int,
    val slidingWindow: Int,
    val ropeBaseLocal: Float,
    val ropeBaseGlobal: Float,
    val kvSharedLayers: Int,
    val layerPattern: List<String>
) {
    /**
     * Returns the layer type at the given layer index.
     * Pattern repeats: ["sliding", "sliding", "sliding", "sliding", "full"]
     */
    public fun getLayerType(layerIdx: Int): LayerType {
        val patternIdx = layerIdx % layerPattern.size
        return when (layerPattern[patternIdx]) {
            "full", "global" -> LayerType.GLOBAL
            else -> LayerType.SLIDING
        }
    }

    /**
     * Returns the RoPE base frequency for the given layer.
     * Local/sliding layers use ropeBaseLocal (10k), global layers use ropeBaseGlobal (1M).
     */
    public fun getRopeBase(layerIdx: Int): Float {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> ropeBaseGlobal
            LayerType.SLIDING -> ropeBaseLocal
        }
    }

    /**
     * Returns the effective attention window size for the given layer.
     * Sliding layers use slidingWindow, global layers use contextLength.
     */
    public fun getEffectiveWindow(layerIdx: Int): Int {
        return when (getLayerType(layerIdx)) {
            LayerType.GLOBAL -> contextLength
            LayerType.SLIDING -> slidingWindow
        }
    }

    /**
     * Returns whether the given layer shares its KV cache with another layer.
     * The last kvSharedLayers layers share their KV cache.
     */
    public fun isKvShared(layerIdx: Int): Boolean {
        return layerIdx >= (blockCount - kvSharedLayers)
    }

    /**
     * Returns the cache layer index for the given layer.
     * Shared layers map to a single cache slot.
     */
    public fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (isKvShared(layerIdx)) {
            blockCount - kvSharedLayers
        } else {
            layerIdx
        }
    }

    /**
     * Returns the feed-forward dimension for the given layer.
     */
    public fun getFeedForwardLength(layerIdx: Int): Int {
        return feedForwardLengths.getOrElse(layerIdx) {
            feedForwardLengths.lastOrNull() ?: (embeddingLength * 4)
        }
    }

    public companion object {
        /** Default layer pattern: 4 sliding + 1 global, repeating */
        public val DEFAULT_LAYER_PATTERN: List<String> = listOf("sliding", "sliding", "sliding", "sliding", "full")

        /** Default sliding window size */
        public const val DEFAULT_SLIDING_WINDOW: Int = 512

        /** Default local RoPE base frequency */
        public const val DEFAULT_ROPE_BASE_LOCAL: Float = 10000f

        /** Default global RoPE base frequency */
        public const val DEFAULT_ROPE_BASE_GLOBAL: Float = 1000000f

        /** Default number of KV shared layers */
        public const val DEFAULT_KV_SHARED_LAYERS: Int = 15
    }
}

/**
 * Type of attention layer in Gemma 3n.
 */
public enum class LayerType {
    /** Local sliding-window attention */
    SLIDING,
    /** Global full-context attention */
    GLOBAL
}
