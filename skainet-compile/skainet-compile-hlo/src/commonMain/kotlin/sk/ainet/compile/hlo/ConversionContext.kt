package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode

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