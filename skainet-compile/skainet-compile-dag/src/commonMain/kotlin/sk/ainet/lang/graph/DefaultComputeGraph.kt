package sk.ainet.lang.graph

import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Default implementation of ComputeGraph with topological sorting and validation
 */
public class DefaultComputeGraph : ComputeGraph {

    private val _nodes = mutableListOf<GraphNode>()
    private val _edges = mutableListOf<GraphEdge>()

    override val nodes: List<GraphNode> get() = _nodes.toList()
    override val edges: List<GraphEdge> get() = _edges.toList()

    override fun addNode(node: GraphNode): GraphNode {
        if (_nodes.any { it.id == node.id }) {
            throw IllegalArgumentException("Node with id '${node.id}' already exists")
        }
        _nodes.add(node)
        return node
    }

    override fun addEdge(edge: GraphEdge): GraphEdge {
        if (_edges.any { it.id == edge.id }) {
            throw IllegalArgumentException("Edge with id '${edge.id}' already exists")
        }

        // Verify that source and destination nodes exist
        if (!_nodes.contains(edge.source)) {
            throw IllegalArgumentException("Source node '${edge.source.id}' not found in graph")
        }
        if (!_nodes.contains(edge.destination)) {
            throw IllegalArgumentException("Destination node '${edge.destination.id}' not found in graph")
        }

        _edges.add(edge)
        return edge
    }

    override fun removeNode(node: GraphNode): Boolean {
        if (!_nodes.contains(node)) return false

        // Remove all edges connected to this node
        _edges.removeAll { it.source == node || it.destination == node }

        return _nodes.remove(node)
    }

    override fun removeEdge(edge: GraphEdge): Boolean {
        return _edges.remove(edge)
    }

    override fun getInputNodes(): List<GraphNode> {
        return _nodes.filter { node ->
            _edges.none { it.destination == node }
        }
    }

    override fun getOutputNodes(): List<GraphNode> {
        return _nodes.filter { node ->
            _edges.none { it.source == node }
        }
    }

    override fun getInputNodes(node: GraphNode): List<GraphNode> {
        return _edges
            .filter { it.destination == node }
            .sortedBy { it.destinationInputIndex }
            .map { it.source }
    }

    override fun getOutputNodes(node: GraphNode): List<GraphNode> {
        return _edges
            .filter { it.source == node }
            .map { it.destination }
    }

    override fun getTopologicalOrder(): List<GraphNode> {
        if (_nodes.isEmpty()) return emptyList()

        // Kahn's algorithm for topological sorting
        val inDegree = mutableMapOf<GraphNode, Int>()
        val queue = mutableListOf<GraphNode>()
        val result = mutableListOf<GraphNode>()

        // Initialize in-degree count for all nodes
        _nodes.forEach { node ->
            inDegree[node] = _edges.count { it.destination == node }
        }

        // Add all nodes with no incoming edges to the queue
        inDegree.forEach { (node, degree) ->
            if (degree == 0) {
                queue.add(node)
            }
        }

        // Process nodes in topological order
        while (queue.isNotEmpty()) {
            val current = queue.removeAt(0)
            result.add(current)

            // For each neighbor of the current node
            getOutputNodes(current).forEach { neighbor ->
                inDegree[neighbor] = inDegree[neighbor]!! - 1
                if (inDegree[neighbor] == 0) {
                    queue.add(neighbor)
                }
            }
        }

        // If result doesn't contain all nodes, there's a cycle
        if (result.size != _nodes.size) {
            throw IllegalStateException("Graph contains cycles - cannot determine topological order")
        }

        return result
    }

    override fun validate(): ValidationResult {
        val errors = mutableListOf<String>()

        // Check for duplicate node IDs
        val nodeIds = _nodes.map { it.id }
        val duplicateNodeIds = nodeIds.groupingBy { it }.eachCount().filter { it.value > 1 }.keys
        if (duplicateNodeIds.isNotEmpty()) {
            errors.add("Duplicate node IDs found: $duplicateNodeIds")
        }

        // Check for duplicate edge IDs
        val edgeIds = _edges.map { it.id }
        val duplicateEdgeIds = edgeIds.groupingBy { it }.eachCount().filter { it.value > 1 }.keys
        if (duplicateEdgeIds.isNotEmpty()) {
            errors.add("Duplicate edge IDs found: $duplicateEdgeIds")
        }

        // Check that all edge nodes exist in the graph
        _edges.forEach { edge ->
            if (!_nodes.contains(edge.source)) {
                errors.add("Edge '${edge.id}' references non-existent source node '${edge.source.id}'")
            }
            if (!_nodes.contains(edge.destination)) {
                errors.add("Edge '${edge.id}' references non-existent destination node '${edge.destination.id}'")
            }
        }

        // Check for cycles by attempting topological sort
        try {
            getTopologicalOrder()
        } catch (_: IllegalStateException) {
            errors.add("Graph contains cycles")
        }

        // Check that edge indices are within bounds of node specs
        _edges.forEach { edge ->
            val srcOutCount = edge.source.outputs.size
            val dstInCount = edge.destination.inputs.size
            if (edge.sourceOutputIndex < 0 || edge.sourceOutputIndex >= srcOutCount) {
                errors.add("Edge '${edge.id}' has invalid sourceOutputIndex=${edge.sourceOutputIndex}; source '${edge.source.id}' provides $srcOutCount outputs")
            }
            if (edge.destinationInputIndex < 0 || edge.destinationInputIndex >= dstInCount) {
                errors.add("Edge '${edge.id}' has invalid destinationInputIndex=${edge.destinationInputIndex}; destination '${edge.destination.id}' expects $dstInCount inputs")
            }
        }

        // Check for orphaned nodes (nodes with no connections)
        val connectedNodes = (_edges.map { it.source } + _edges.map { it.destination }).toSet()
        val orphanedNodes = _nodes.filter { !connectedNodes.contains(it) }
        if (orphanedNodes.isNotEmpty() && _nodes.size > 1) {
            errors.add("Orphaned nodes found (no connections): ${orphanedNodes.map { it.id }}")
        }

        return if (errors.isEmpty()) {
            ValidationResult.Valid
        } else {
            ValidationResult.Invalid(errors)
        }
    }

    override fun copy(): ComputeGraph {
        val copy = DefaultComputeGraph()

        // Copy nodes
        _nodes.forEach { node ->
            copy.addNode(node.copy())
        }

        // Copy edges (need to find the copied nodes)
        _edges.forEach { edge ->
            val copiedSource = copy._nodes.first { it.id == edge.source.id }
            val copiedDestination = copy._nodes.first { it.id == edge.destination.id }

            copy.addEdge(edge.copy(
                source = copiedSource,
                destination = copiedDestination
            ))
        }

        return copy
    }

    override fun clear() {
        _edges.clear()
        _nodes.clear()
    }

    override fun toString(): String {
        return "DefaultComputeGraph(nodes=${_nodes.size}, edges=${_edges.size})"
    }
}