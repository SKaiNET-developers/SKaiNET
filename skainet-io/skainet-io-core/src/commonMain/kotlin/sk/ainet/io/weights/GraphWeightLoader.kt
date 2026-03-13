package sk.ainet.io.weights

import sk.ainet.lang.dag.GraphNodeDefinition
import sk.ainet.lang.dag.GraphProgram
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Loads weights from model files directly into a [GraphProgram]'s parameter nodes.
 *
 * This is the post-trace weight loading path: after a model is traced into a DAG,
 * parameter nodes need to be bound to actual tensor data. Unlike [WeightMapper]
 * (which operates on `Module<T, V>` trees), this class operates on the immutable
 * [GraphProgram] representation and returns a map of parameter node IDs to loaded tensors.
 *
 * Usage:
 * ```kotlin
 * val loader = GraphWeightLoader(LlamaGGUFNameResolver())
 * val weights = loader.load(program, weightTensors)
 * // weights: Map<String, Tensor<T, V>> keyed by parameter node ID
 * ```
 *
 * @param resolver Translates parameter node IDs (derived from DSL module paths) to
 *                 tensor names in the model file format.
 */
public class GraphWeightLoader(
    private val resolver: WeightNameResolver
) {

    /**
     * Result of loading weights into graph parameter nodes.
     *
     * @property weights Map of parameter node ID to loaded tensor
     * @property mapped Number of parameters that were successfully loaded
     * @property total Total number of parameter nodes in the graph
     * @property missingParams Parameter node IDs for which no tensor was found
     * @property unusedTensors Tensor names from the source that were not consumed
     */
    public data class GraphLoadResult<T : DType, V>(
        val weights: Map<String, Tensor<T, V>>,
        val mapped: Int,
        val total: Int,
        val missingParams: List<String>,
        val unusedTensors: List<String>
    )

    /**
     * Load weights from the given [tensors] into the parameter nodes of [program].
     *
     * Parameter nodes are identified by having:
     * - An [InputOperation][sk.ainet.lang.tensor.ops.InputOperation] with `kind = "parameter"`, or
     * - An attribute `role = "parameter"`
     *
     * For each parameter node, the resolver translates the node's ID into a model-file
     * tensor name, which is looked up in [tensors].
     *
     * @param program The traced graph program containing parameter nodes
     * @param tensors The weight tensors loaded from a model file
     * @return A [GraphLoadResult] containing the loaded weights and diagnostics
     */
    public fun <T : DType, V> load(
        program: GraphProgram,
        tensors: List<WeightTensor<T, V>>
    ): GraphLoadResult<T, V> {
        val paramNodes = program.nodes.filter { isParameterNode(it) }
        val tensorsByName = tensors.associateBy { it.name }
        val used = mutableSetOf<String>()
        val weights = mutableMapOf<String, Tensor<T, V>>()
        val missing = mutableListOf<String>()

        for (param in paramNodes) {
            // Extract a module-path-like name from the node ID
            // Node IDs from tracing look like: "input_t5", "param_blk.0.attn.q_proj.weight"
            val (modulePath, paramName) = splitParamNodeId(param.id)
            val resolvedName = resolver.resolve(modulePath, paramName)

            val tensor = if (resolvedName != null) {
                tensorsByName[resolvedName]
            } else {
                // Fallback: try direct ID match
                tensorsByName[param.id]
                    ?: tensorsByName[modulePath]
                    ?: tensorsByName[paramName]
            }

            if (tensor != null) {
                // Shape validation
                val expectedShape = param.outputs.firstOrNull()?.shape
                if (expectedShape != null && expectedShape != tensor.shape) {
                    missing.add("${param.id}: shape mismatch expected=$expectedShape actual=${tensor.shape}")
                    continue
                }
                weights[param.id] = tensor.tensor
                used.add(tensor.name)
            } else {
                missing.add(param.id)
            }
        }

        val unused = tensors.filter { it.name !in used }.map { it.name }

        return GraphLoadResult(
            weights = weights,
            mapped = weights.size,
            total = paramNodes.size,
            missingParams = missing,
            unusedTensors = unused
        )
    }

    /**
     * Validate that all parameter nodes were loaded.
     * Throws [IllegalArgumentException] if any parameters are missing.
     */
    public fun <T : DType, V> validateAllLoaded(result: GraphLoadResult<T, V>) {
        require(result.mapped == result.total) {
            buildString {
                appendLine("Only loaded ${result.mapped}/${result.total} graph parameters; aborting.")
                if (result.missingParams.isNotEmpty()) {
                    appendLine("Missing parameters (${result.missingParams.size}):")
                    result.missingParams.take(10).forEach { appendLine("  - $it") }
                    if (result.missingParams.size > 10) appendLine("  - ... and ${result.missingParams.size - 10} more")
                }
                if (result.unusedTensors.isNotEmpty()) {
                    appendLine("Unused tensors (${result.unusedTensors.size}):")
                    result.unusedTensors.take(10).forEach { appendLine("  - $it") }
                    if (result.unusedTensors.size > 10) appendLine("  - ... and ${result.unusedTensors.size - 10} more")
                }
            }.trim()
        }
    }

    private fun isParameterNode(node: GraphNodeDefinition): Boolean {
        val op = node.operation
        return (op.name == "input" && op.parameters["kind"] == "parameter") ||
            node.attributes["role"] == "parameter"
    }

    private fun splitParamNodeId(nodeId: String): Pair<String, String> {
        // Strip common prefixes from trace conversion
        val cleaned = nodeId
            .removePrefix("input_")
            .removePrefix("param_")

        // Split into module path and parameter name
        // e.g. "blk.0.attn.q_proj.weight" → ("blk.0.attn", "q_proj.weight")
        val lastDot = cleaned.lastIndexOf('.')
        val secondLastDot = if (lastDot > 0) cleaned.lastIndexOf('.', lastDot - 1) else -1

        return if (secondLastDot >= 0) {
            val modulePath = cleaned.substring(0, secondLastDot)
            val paramName = cleaned.substring(secondLastDot + 1)
            modulePath to paramName
        } else {
            cleaned to cleaned
        }
    }
}
