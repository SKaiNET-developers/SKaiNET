package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Facade for loading BERT models from any format via the generic [ParametersLoader] interface.
 *
 * Follows the [LlamaIngestion] pattern but is IO-format agnostic — the caller provides
 * a [ParametersLoader] which can be backed by SafeTensors, GGUF, ONNX, etc.
 */
public class BertIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: BertModelConfig
) {
    /**
     * Load BERT weights from a generic [ParametersLoader].
     * Works with any format (SafeTensors, GGUF, ONNX) as long as the tensor names
     * follow HuggingFace conventions.
     */
    public suspend fun load(loader: ParametersLoader): BertRuntimeWeights<T> {
        return loadBertWeights(loader, ctx, dtype, config)
    }
}
