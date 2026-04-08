@file:Suppress("unused")

package sk.ainet.lang.tensor.ops.turboquant

import sk.ainet.lang.tensor.storage.*

/**
 * TurboQuant integration guide for skainet-transformers.
 *
 * TurboQuant compresses the KV cache at **runtime** — no model retraining
 * or weight re-quantization needed. Any model (LLaMA, Mistral, Gemma,
 * Qwen, etc.) benefits immediately.
 *
 * ## What TurboQuant does
 *
 * During autoregressive inference, the KV cache grows linearly with
 * sequence length and dominates memory usage. TurboQuant compresses
 * K/V projections on write and decompresses on read:
 *
 * - **4-bit (balanced)**: ~8x compression vs FP32
 * - **3-bit (experimental-max)**: ~10x compression
 * - **safe-lowbit**: Q8_0 keys + 4-bit values (conservative)
 *
 * ## Quick start
 *
 * ### 1. One-line cache creation
 *
 * ```kotlin
 * // Replace your existing KV cache with TurboQuant:
 * val cache = KvCacheStore.turboQuant(
 *     preset = "balanced",
 *     numLayers = 32,
 *     numHeads = 32,
 *     headDim = 128,
 *     maxSeqLen = 4096
 * )
 * ```
 *
 * ### 2. Use in attention layer
 *
 * ```kotlin
 * class MultiHeadAttention(
 *     val numHeads: Int,
 *     val headDim: Int,
 *     val cache: KvCacheStore
 * ) {
 *     private val bridge = CompressedKvAttention(cache)
 *
 *     fun forward(query: FloatArray, key: FloatArray, value: FloatArray, layer: Int): FloatArray {
 *         // Store K/V (compressed automatically)
 *         bridge.storeKeyValue(layer, key, value)
 *
 *         // Read for attention (decompressed automatically)
 *         val cachedKeys = bridge.loadKeysForAttention(layer)
 *         val cachedValues = bridge.loadValuesForAttention(layer)
 *
 *         // Pass to scaledDotProductAttention as usual
 *         return computeAttention(query, cachedKeys, cachedValues)
 *     }
 * }
 * ```
 *
 * ### 3. Annotate layers (optional)
 *
 * ```kotlin
 * @KvCache(preset = "balanced")
 * class SelfAttention(...) { ... }
 *
 * // Resolve at model init:
 * val cache = KvCacheAnnotationResolver.resolve(
 *     preset = "balanced",
 *     numLayers = config.numLayers,
 *     numHeads = config.numKVHeads,
 *     headDim = config.headDim,
 *     maxSeqLen = config.maxSeqLen
 * )
 * ```
 *
 * ### 4. Monitor compression
 *
 * ```kotlin
 * val report = cache.memoryReport()
 * println("Compression: ${report.compressionRatio}x")
 * println("KV cache: ${report.totalPhysicalBytes / 1024 / 1024} MB")
 * println("Utilization: ${(report.utilizationRatio * 100).toInt()}%")
 * ```
 *
 * ## Preset selection guide
 *
 * | Preset | Key bits | Value bits | Compression | Quality | Use case |
 * |--------|----------|------------|-------------|---------|----------|
 * | safe-lowbit | 8 (Q8_0) | 4 (TQ) | ~4-6x | Best | Production, quality-sensitive |
 * | balanced | 4 (TQ) | 4 (TQ) | ~8x | Good | General purpose, long context |
 * | experimental-max | 3 (TQ) | 3 (TQ) | ~10x | Fair | Memory-constrained, very long context |
 *
 * ## Model compatibility
 *
 * TurboQuant works with **any model** regardless of:
 * - Weight quantization format (GGUF Q4_K, Q8_0, FP16, etc.)
 * - Architecture (LLaMA, Mistral, Gemma, Qwen, BERT)
 * - Model size (1B to 70B+)
 * - Age (works with older models too)
 *
 * The model weights are unchanged — only the KV cache storage is compressed.
 */
public object TurboQuantUsage {

    /**
     * Example: Create a balanced TurboQuant cache for a LLaMA-style model.
     *
     * This is a compilable reference showing the full integration pattern.
     */
    public fun exampleLlamaCache(): KvCacheStore {
        // LLaMA-7B dimensions
        val numLayers = 32
        val numHeads = 32    // or numKVHeads for GQA models
        val headDim = 128
        val maxSeqLen = 4096

        // One-line creation:
        return KvCacheStore.turboQuant("balanced", numLayers, numHeads, headDim, maxSeqLen)
    }

    /**
     * Example: Asymmetric K/V compression (8-bit keys, 4-bit values).
     */
    public fun exampleAsymmetricCache(): KvCacheStore {
        return KvCacheStore.turboQuant(
            numLayers = 32,
            numHeads = 8,      // GQA: 8 KV heads
            headDim = 128,
            maxSeqLen = 8192,
            keyBits = 8,       // High precision for keys
            valueBits = 4      // Lower precision for values
        )
    }

    /**
     * Example: Full generation loop with TurboQuant KV cache.
     *
     * Shows how TurboQuant integrates into token-by-token inference.
     */
    public fun exampleGenerationLoop() {
        val numLayers = 4
        val numHeads = 4
        val headDim = 64
        val maxSeqLen = 128

        // Create compressed cache
        val cache = KvCacheStore.turboQuant("balanced", numLayers, numHeads, headDim, maxSeqLen)
        val bridge = CompressedKvAttention(cache)

        // Simulate generation of 10 tokens
        for (token in 0 until 10) {
            for (layer in 0 until numLayers) {
                // Simulate K/V projections (in real code, this comes from linear layers)
                val key = FloatArray(numHeads * headDim) { it.toFloat() / (numHeads * headDim) }
                val value = FloatArray(numHeads * headDim) { -it.toFloat() / (numHeads * headDim) }

                // Store with TurboQuant compression (transparent)
                bridge.storeKeyValue(layer, key, value)

                // Read decompressed K/V for attention
                val cachedKeys = bridge.loadKeysForAttention(layer)
                val cachedValues = bridge.loadValuesForAttention(layer)

                // ... pass to scaledDotProductAttention ...
            }
        }

        // Check compression
        val report = cache.memoryReport()
        val savedBytes = report.totalLogicalBytes - report.totalPhysicalBytes
        // With balanced preset: ~8x compression
    }
}
