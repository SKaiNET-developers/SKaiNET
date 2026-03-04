package sk.ainet.models.gemma

import sk.ainet.io.gguf.TensorNameMapper

/**
 * GGUF tensor naming for Gemma 3n models.
 *
 * Gemma 3n uses the same `blk.N.*` base naming as LLaMA in GGUF format,
 * but with different norm names (`attn_norm` vs `input_layernorm`) and
 * additional per-layer embedding tensors.
 */
public object Gemma3nGgufTensorNames : TensorNameMapper {
    override fun tokenEmbedding(): String = Gemma3nTensorNames.TOKEN_EMBEDDINGS
    override fun outputNorm(): String = Gemma3nTensorNames.OUTPUT_NORM
    override fun outputWeight(): String = Gemma3nTensorNames.OUTPUT_WEIGHT

    override fun layerAttnNorm(layer: Int): String = Gemma3nTensorNames.inputLayernorm(layer)
    override fun layerAttnQ(layer: Int): String = Gemma3nTensorNames.attnQ(layer)
    override fun layerAttnK(layer: Int): String = Gemma3nTensorNames.attnK(layer)
    override fun layerAttnV(layer: Int): String = Gemma3nTensorNames.attnV(layer)
    override fun layerAttnO(layer: Int): String = Gemma3nTensorNames.attnOut(layer)

    override fun layerFfnNorm(layer: Int): String = Gemma3nTensorNames.postAttentionLayernorm(layer)
    override fun layerFfnGate(layer: Int): String = Gemma3nTensorNames.ffnGate(layer)
    override fun layerFfnUp(layer: Int): String = Gemma3nTensorNames.ffnUp(layer)
    override fun layerFfnDown(layer: Int): String = Gemma3nTensorNames.ffnDown(layer)
}
