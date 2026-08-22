package sk.ainet.lang.nn.topology

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.TensorIdBearer

/**
 * Result of [assignTensorIds]: every parameter tensor of the tree keyed by its [TensorId], and the
 * ids whose tensor could not carry the id (not a [TensorIdBearer]) — they are still in [tensors].
 */
public data class TensorIdAssignment(
    val tensors: Map<TensorId, Tensor<*, *>>,
    val notCarried: List<TensorId>,
) {
    public operator fun get(id: TensorId): Tensor<*, *>? = tensors[id]
    public operator fun get(canonical: String): Tensor<*, *>? = tensors[TensorId.parse(canonical)]
}

/**
 * Assign a [TensorId] to every parameter tensor in this module tree, derived from the module
 * structure: the id's module path is the chain of module names from [root] down (the same segments
 * [bindPaths] uses, so `id.legacyPath()` equals the node's `path`), and its parameter is the
 * parameter's short name (`weight` for a parameter registered as `"<module>.weight"`).
 *
 * Idempotent — running it twice yields the same ids — and free of side effects on tensors that
 * cannot carry an id (reported in [TensorIdAssignment.notCarried]). No runtime code path changes:
 * an id is metadata on the parameter tensor.
 *
 * ```
 * val ids = model.assignTensorIds("model")
 * ids["model.layers.blk.0.attn.weight"]   // the tensor
 * ```
 */
public fun ModuleNode.assignTensorIds(root: String = name): TensorIdAssignment {
    val tensors = LinkedHashMap<TensorId, Tensor<*, *>>()
    val notCarried = ArrayList<TensorId>()
    fun visit(node: ModuleNode, segments: List<String>) {
        for (p in node.params) {
            val id = TensorId(segments, parameterShortName(node, p.name))
            val t = p.value
            tensors[id] = t
            if (t is TensorIdBearer) t.id = id else notCarried += id
        }
        for (child in node.children) {
            val seg = child.name.ifEmpty { child.id }
            visit(child, if (seg.isEmpty()) segments else segments + seg)
        }
    }
    visit(this, if (root.isEmpty()) emptyList() else listOf(root))
    return TensorIdAssignment(tensors, notCarried)
}

/** `"<moduleName>.weight"` → `weight`; a name without the module prefix is returned unchanged. */
internal fun parameterShortName(node: ModuleNode, parameterName: String): String {
    val prefix = node.name + "."
    return when {
        node.name.isNotEmpty() && parameterName.startsWith(prefix) && parameterName.length > prefix.length ->
            parameterName.substring(prefix.length)
        else -> parameterName
    }
}
