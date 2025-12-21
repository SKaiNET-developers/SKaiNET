package sk.ainet.apps.knanogpt.llm

import sk.ainet.apps.knanogpt.transformer.TransformerConfig
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

/**
 * Placeholder bigram language model wired to the current Module API.
 * TODO: implement token embedding and sampling once the data pipeline is in place.
 */
class BigramLanguageModel(
    private val config: TransformerConfig,
    override val name: String = "BigramLanguageModel"
) : Module<FP16, Float>() {

    override val modules: List<Module<FP16, Float>>
        get() = emptyList()

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> {
        throw NotImplementedError("BigramLanguageModel is not implemented yet.")
    }
}
