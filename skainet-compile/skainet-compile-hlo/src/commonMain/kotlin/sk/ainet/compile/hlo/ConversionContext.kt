package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.tensorEncoding

/**
 * Context object for maintaining state during StableHLO conversion.
 * 
 * This class manages SSA value names, type mapping, and MLIR code generation
 * during the conversion process from ComputeGraph to StableHLO.
 */
public class ConversionContext(
    private val typeMapper: TypeMapper,
    private var graph: ComputeGraph? = null
) {
    private val valueNames = mutableMapOf<String, String>()
    private val stringBuilder = StringBuilder()
    private var tempCounter = 0
    
    /**
     * Get the SSA value name for a node ID
     */
    public fun getValueName(nodeId: String): String? = valueNames[nodeId]
    
    /**
     * Set the SSA value name for a node ID
     */
    public fun setValueName(nodeId: String, valueName: String) {
        valueNames[nodeId] = valueName
    }
    
    /**
     * Generate the next temporary SSA value name
     */
    public fun nextTempValue(): String = "%v${tempCounter++}"
    
    /**
     * Emit a line of MLIR code with proper indentation
     */
    public fun emitLine(line: String) {
        stringBuilder.appendLine(line)
    }
    
    /**
     * Emit an operation with proper indentation
     */
    public fun emitOperation(operation: String) {
        stringBuilder.appendLine("    $operation")
    }
    
    /**
     * Emit a comment with proper indentation
     */
    public fun emitComment(comment: String) {
        stringBuilder.appendLine("    // $comment")
    }

    /**
     * Emit a `tensor_encoding` diagnostic comment when [spec] carries a
     * non-null `tensorEncoding` (set via [sk.ainet.lang.tensor.ops.withTensorEncoding]).
     *
     * The emitted line has the shape:
     *
     * ```mlir
     *     // tensor_encoding: role=<role> index=<i> name=<spec.name> encoding=<enc.name>
     * ```
     *
     * MLIR tools ignore comments but text round-trips preserve them, so
     * this is the cheapest way to keep SKaiNET's quantization metadata
     * visible through the StableHLO emit boundary until a structured
     * attribute or quant-dialect lowering lands. Emits nothing when the
     * spec has no encoding — a `null` [sk.ainet.lang.tensor.storage.TensorEncoding]
     * is the unknown / not-carried state, intentionally distinct from
     * `TensorEncoding.Dense`.
     *
     * @param role Logical slot the spec occupies for the node being
     *     emitted (e.g. `"input"` for function arguments, `"result"` for
     *     node outputs). Free-form so individual converters can use
     *     finer-grained tags if they call this helper directly.
     * @param index Positional index of the spec within its role, e.g.
     *     the output port index for a multi-result node.
     */
    public fun emitEncodingAnnotation(role: String, index: Int, spec: TensorSpec) {
        val encoding = spec.tensorEncoding ?: return
        emitComment(
            "tensor_encoding: role=$role index=$index name=${spec.name} encoding=${encoding.name}"
        )
    }
    
    /**
     * Get the complete generated MLIR content
     */
    public fun getContent(): String = stringBuilder.toString()
    
    /**
     * Set the graph reference for node lookups
     */
    public fun setGraph(graph: ComputeGraph) {
        this.graph = graph
    }
    
    /**
     * Get input nodes for a given node from the graph
     */
    public fun getInputNodes(node: GraphNode): List<GraphNode> {
        return graph?.getInputNodes(node) ?: emptyList()
    }
    
    /**
     * Get the type mapper instance
     */
    public fun getTypeMapper(): TypeMapper = typeMapper
    
    /**
     * Clear all state (useful for testing)
     */
    public fun clear() {
        valueNames.clear()
        stringBuilder.clear()
        tempCounter = 0
    }
}