package sk.ainet.lang.nn.layers

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Adapter that wraps an [Embedding] (DualModule<Int32, T, V>) as a [Module<T, V>]
 * so it can be used in sequential `Module<T, V>` pipelines (e.g., the network DSL).
 *
 * Input tensors are expected to contain integer token IDs stored as floats.
 * The adapter delegates to [Embedding.forwardAny] which handles the conversion.
 */
public class EmbeddingAdapter<T : DType, V>(
    public val embedding: Embedding<T, V>
) : Module<T, V>(), ModuleParameters<T, V> {

    override val name: String get() = embedding.name

    public val numEmbeddings: Int get() = embedding.numEmbeddings
    public val embeddingDim: Int get() = embedding.embeddingDim

    override val modules: List<Module<T, V>> get() = emptyList()

    override val params: List<ModuleParameter<T, V>>
        get() = embedding.params

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // The input tensor contains token IDs as float values.
        // Embedding.forwardAny handles the conversion from arbitrary DType to Int32 indices.
        return embedding.forwardAny(input, ctx, strict = false)
    }
}
