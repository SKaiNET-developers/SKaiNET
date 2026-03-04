package sk.ainet.models.llama

import sk.ainet.io.gguf.TensorNameMapper

/**
 * Standard GGUF tensor naming for LLaMA-family models.
 *
 * Uses the canonical `blk.N.*` naming scheme from llama.cpp GGUF files.
 * This is also the default scheme for Mistral, Qwen, and other LLaMA derivatives.
 */
public object LlamaGgufTensorNames : TensorNameMapper {
    override fun tokenEmbedding(): String = LlamaTensorNames.TOKEN_EMBEDDINGS
    override fun outputNorm(): String = LlamaTensorNames.OUTPUT_NORM
    override fun outputWeight(): String = LlamaTensorNames.OUTPUT_WEIGHT

    override fun layerAttnNorm(layer: Int): String = LlamaTensorNames.attnNorm(layer)
    override fun layerAttnQ(layer: Int): String = LlamaTensorNames.attnQ(layer)
    override fun layerAttnK(layer: Int): String = LlamaTensorNames.attnK(layer)
    override fun layerAttnV(layer: Int): String = LlamaTensorNames.attnV(layer)
    override fun layerAttnO(layer: Int): String = LlamaTensorNames.attnOut(layer)

    override fun layerFfnNorm(layer: Int): String = LlamaTensorNames.ffnNorm(layer)
    override fun layerFfnGate(layer: Int): String = LlamaTensorNames.ffnGate(layer)
    override fun layerFfnUp(layer: Int): String = LlamaTensorNames.ffnUp(layer)
    override fun layerFfnDown(layer: Int): String = LlamaTensorNames.ffnDown(layer)
}
