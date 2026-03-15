package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Evaluates binary operations whose inputs are all constant nodes at compile time.
 *
 * A node is considered constant when `operation.name == "constant"` and its values
 * are stored in `operation.parameters["values"]`.
 *
 * The pass walks in topological order so that folded results propagate forward
 * within a single application. Use fixed-point iteration in the pipeline for
 * multi-level folding chains.
 */
public class ConstantFoldingPass : GraphOptimizationPass {
    override val name: String = "constant-folding"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val topoOrder = graph.getTopologicalOrder()
        val diagnostics = mutableListOf<String>()

        // Constant value cache: nodeId → flat float list
        val constantValues = mutableMapOf<String, List<Float>>()

        // Seed with existing constants
        for (node in topoOrder) {
            if (node.operation.name == "constant") {
                extractFloatValues(node)?.let { constantValues[node.id] = it }
            }
        }

        // Identify foldable binary ops
        val foldedNodes = mutableMapOf<String, Pair<GraphNode, List<Float>>>() // nodeId → (replacement constant node, values)

        for (node in topoOrder) {
            if (node.id in constantValues) continue // already a constant

            val opName = node.operation.name
            if (opName !in FOLDABLE_OPS) continue

            // All inputs must be constants
            val inputNodes = graph.getInputNodes(node)
            if (inputNodes.isEmpty()) continue

            val inputValues = inputNodes.map { constantValues[it.id] ?: return@map null }
            if (inputValues.any { it == null }) continue

            @Suppress("UNCHECKED_CAST")
            val typedInputs = inputValues as List<List<Float>>

            // Binary ops need exactly 2 inputs of equal size
            if (typedInputs.size != 2) continue
            val lhs = typedInputs[0]
            val rhs = typedInputs[1]
            if (lhs.size != rhs.size) continue

            val result = when (opName) {
                "add" -> lhs.zip(rhs) { a, b -> a + b }
                "multiply" -> lhs.zip(rhs) { a, b -> a * b }
                "subtract" -> lhs.zip(rhs) { a, b -> a - b }
                "divide" -> {
                    if (rhs.any { it == 0.0f }) continue // skip division by zero
                    lhs.zip(rhs) { a, b -> a / b }
                }
                else -> continue
            }

            // Create a replacement constant node
            val constOp = GenericOperation(
                name = "constant",
                parameters = mapOf("values" to result),
                type = "constant"
            )
            val replacement = node.copy(operation = constOp)
            foldedNodes[node.id] = replacement to result
            constantValues[node.id] = result
            diagnostics.add("Folded ${opName}: ${node.id} (${lhs.size} elements)")
        }

        if (foldedNodes.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Rebuild graph, substituting folded nodes
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            val replacement = foldedNodes[node.id]?.first
            val toAdd = (replacement ?: node).copy()
            newGraph.addNode(toAdd)
            nodeMap[toAdd.id] = toAdd
        }

        for (edge in graph.edges) {
            val src = nodeMap[edge.source.id] ?: continue
            val dst = nodeMap[edge.destination.id] ?: continue
            // If the destination was folded, we still keep edges for downstream consumers.
            // Edges *into* folded nodes become dead; DCE will clean them up.
            newGraph.addEdge(edge.copy(source = src, destination = dst))
        }

        return GraphOptimizationResult(
            graph = newGraph,
            changed = true,
            diagnostics = diagnostics
        )
    }

    private companion object {
        val FOLDABLE_OPS = setOf("add", "multiply", "subtract", "divide")

        @Suppress("UNCHECKED_CAST")
        fun extractFloatValues(node: GraphNode): List<Float>? {
            val values = node.operation.parameters["values"] ?: return null
            return when (values) {
                is List<*> -> (values as? List<Number>)?.map { it.toFloat() }
                is FloatArray -> values.toList()
                is DoubleArray -> values.map { it.toFloat() }
                else -> null
            }
        }
    }
}
