package sk.ainet.lang.graph.dsl

import sk.ainet.lang.dag.GraphProgram
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode

/**
 * Lower a symbolic [GraphProgram] (produced by the DAG DSL) into a concrete [ComputeGraph].
 *
 * Node and edge identifiers follow the same deterministic pattern as trace-based wiring:
 *  - node ids are preserved from the DSL (defaults are n<seq>_<op>),
 *  - edge ids are derived from endpoints and port indices.
 */
public fun GraphProgram.toComputeGraph(into: ComputeGraph = DefaultComputeGraph()): ComputeGraph {
    val graph = into
    val nodeById = mutableMapOf<String, GraphNode>()

    // First register all nodes.
    nodes.forEach { def ->
        val metadata = def.attributes
            .filterValues { it != null }
            .mapValues { it.value as Any }
        val node = GraphNode(
            id = def.id,
            operation = def.operation,
            inputs = def.inputs.map { it.spec },
            outputs = def.outputs.map { it.spec },
            metadata = metadata
        )
        graph.addNode(node)
        nodeById[def.id] = node
    }

    // Then wire edges based on recorded producer ids.
    nodes.forEach { def ->
        val dstNode = nodeById[def.id] ?: return@forEach
        def.inputs.forEachIndexed { inputIdx, value ->
            val srcNode = nodeById[value.nodeId] ?: return@forEachIndexed
            val edgeId = "e_${srcNode.id}_${value.outputIndex}__${dstNode.id}_$inputIdx"
            graph.addEdge(
                GraphEdge(
                    id = edgeId,
                    source = srcNode,
                    destination = dstNode,
                    sourceOutputIndex = value.outputIndex,
                    destinationInputIndex = inputIdx,
                    tensorSpec = value.spec
                )
            )
        }
    }

    return graph
}
