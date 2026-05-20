package sk.ainet.lang.graph

import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Minimal production ComputeGraph implementation for tape-based recording path.
 * This mirrors the behavior of the test DefaultComputeGraph but is simplified.
 */
public class SimpleComputeGraph(
    nodesInit: List<GraphNode> = emptyList(),
    edgesInit: List<GraphEdge> = emptyList()
) : ComputeGraph {

    private val _nodes = mutableListOf<GraphNode>().apply { addAll(nodesInit) }
    private val _edges = mutableListOf<GraphEdge>().apply { addAll(edgesInit) }

    override val nodes: List<GraphNode> get() = _nodes.toList()
    override val edges: List<GraphEdge> get() = _edges.toList()

    override fun addNode(node: GraphNode): GraphNode {
        require(_nodes.none { it.id == node.id }) { "Node with id '${node.id}' already exists" }
        _nodes += node
        return node
    }

    override fun addEdge(edge: GraphEdge): GraphEdge {
        require(_edges.none { it.id == edge.id }) { "Edge with id '${edge.id}' already exists" }
        require(_nodes.contains(edge.source)) { "Source node '${edge.source.id}' not found in graph" }
        require(_nodes.contains(edge.destination)) { "Destination node '${edge.destination.id}' not found in graph" }
        _edges += edge
        return edge
    }

    override fun removeNode(node: GraphNode): Boolean {
        if (!_nodes.remove(node)) return false
        _edges.removeAll { it.source == node || it.destination == node }
        return true
    }

    override fun removeEdge(edge: GraphEdge): Boolean = _edges.remove(edge)

    override fun getInputNodes(): List<GraphNode> = _nodes.filter { n -> _edges.none { it.destination == n } }
    override fun getOutputNodes(): List<GraphNode> = _nodes.filter { n -> _edges.none { it.source == n } }

    override fun getInputNodes(node: GraphNode): List<GraphNode> =
        _edges.filter { it.destination == node }.sortedBy { it.destinationInputIndex }.map { it.source }
    override fun getOutputNodes(node: GraphNode): List<GraphNode> = _edges.filter { it.source == node }.map { it.destination }

    override fun getTopologicalOrder(): List<GraphNode> {
        if (_nodes.isEmpty()) return emptyList()
        val inDegree = mutableMapOf<GraphNode, Int>()
        _nodes.forEach { n -> inDegree[n] = _edges.count { it.destination == n } }
        val q = ArrayDeque<GraphNode>()
        inDegree.filter { it.value == 0 }.forEach { q.addLast(it.key) }
        val result = mutableListOf<GraphNode>()
        while (q.isNotEmpty()) {
            val n = q.removeFirst()
            result += n
            getOutputNodes(n).forEach { m ->
                inDegree[m] = (inDegree[m] ?: 0) - 1
                if (inDegree[m] == 0) q.addLast(m)
            }
        }
        return result
    }

    override fun validate(): ValidationResult {
        val order = getTopologicalOrder()
        return if (order.size == _nodes.size) ValidationResult.Valid
        else ValidationResult.Invalid(listOf("Graph contains cycles or disconnected issues"))
    }

    override fun copy(): ComputeGraph = SimpleComputeGraph(nodes, edges)
    override fun clear() { _nodes.clear(); _edges.clear() }
}
