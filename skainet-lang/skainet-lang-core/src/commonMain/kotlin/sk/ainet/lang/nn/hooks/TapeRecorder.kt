package sk.ainet.lang.nn.hooks

import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Simple tape recorder that collects forward pass records.
 * Zero-cost when not installed in the ExecutionContext (hooks == null).
 */
public class TapeRecorder : ForwardHooks {

    public data class Entry(
        val moduleId: String,
        val moduleName: String,
        val modulePath: String?,
        val inputSpec: TensorSpec?,
        val outputSpec: TensorSpec?,
        val paramRefs: List<String>,
    )

    private val _tape = mutableListOf<Entry>()
    public val tape: List<Entry> get() = _tape

    // We temporarily store the input spec per module during a begin/end pair
    private val pendingInputs = mutableMapOf<ModuleNode, TensorSpec?>()

    override fun onForwardBegin(module: ModuleNode, input: Any) {
        val spec = input.toTensorSpec()
        pendingInputs[module] = spec
    }

    override fun onForwardEnd(module: ModuleNode, input: Any, output: Any) {
        val inSpec = pendingInputs.remove(module)
        val outSpec = output.toTensorSpec()
        val params: List<String> = module.params.map { it.asRefString() }
        _tape += Entry(
            moduleId = module.id,
            moduleName = module.name,
            modulePath = module.path,
            inputSpec = inSpec,
            outputSpec = outSpec,
            paramRefs = params,
        )
    }
}

private fun ModuleParameter<*, *>.asRefString(): String = "$name@${'$'}{value.shape}:${'$'}{value.dtype.simpleName}"

private fun Any.toTensorSpec(): TensorSpec? = when (this) {
    is Tensor<*, *> -> TensorSpec(
        name = "",
        shape = this.shape.dimensions.toList(),
        dtype = this.dtype.simpleName ?: "Unknown",
        requiresGrad = this.requiresGrad,
        metadata = emptyMap()
    )
    else -> null
}
