package sk.ainet.io.gguf

/**
 * Abstracts the mapping from logical tensor roles to GGUF tensor name strings.
 *
 * Different model families (LLaMA, Gemma, Mistral, Qwen) share the same
 * transformer architecture but use different naming conventions for their
 * weight tensors. This interface provides a single point of indirection
 * so that a weight loader can work with any naming scheme.
 *
 * Implement this interface for each model family and pass it to the
 * weight loader to enable loading models with non-standard tensor names.
 */
public interface TensorNameMapper {
    public fun tokenEmbedding(): String
    public fun outputNorm(): String
    public fun outputWeight(): String

    public fun layerAttnNorm(layer: Int): String
    public fun layerAttnQ(layer: Int): String
    public fun layerAttnK(layer: Int): String
    public fun layerAttnV(layer: Int): String
    public fun layerAttnO(layer: Int): String

    public fun layerFfnNorm(layer: Int): String
    public fun layerFfnGate(layer: Int): String
    public fun layerFfnUp(layer: Int): String
    public fun layerFfnDown(layer: Int): String
}
