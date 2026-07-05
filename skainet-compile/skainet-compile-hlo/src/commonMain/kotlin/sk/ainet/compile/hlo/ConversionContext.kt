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
public class ConversionContext @kotlin.jvm.JvmOverloads constructor(
    private val typeMapper: TypeMapper,
    private var graph: ComputeGraph? = null,
    /**
     * Governs whether constant tensors are inlined as `dense<...>` or
     * lifted into `util.global` module declarations. Default
     * [ConstantMaterializationPolicy.InlineAlways] preserves historical
     * behavior for every caller that constructs a context without
     * naming a policy — the external path is strictly opt-in.
     * See issue #523 for the architecture context.
     */
    public val materializationPolicy: ConstantMaterializationPolicy =
        ConstantMaterializationPolicy.InlineAlways,
    /**
     * Selected compile target (iree device id, e.g. "torq"); `null` = target-agnostic.
     * Threaded through so per-target emit decisions (e.g. op granularity) are possible
     * without any hardware knowledge in the shared emitter.
     */
    public val target: String? = null,
    /**
     * Per-target op-granularity policy (fused vs decomposed emission). Resolved by the
     * caller from the [sk.ainet.compile.opt.TargetOptimizers] registry and passed in;
     * `null` = decompose everything (portable default). The emitter only *reads* it.
     */
    public val granularity: sk.ainet.compile.target.OpGranularityPolicy? = null
) {
    private val valueNames = mutableMapOf<String, String>()
    private val valueTypes = mutableMapOf<String, String>()
    private val stringBuilder = StringBuilder()
    private val moduleDeclarationsBuilder = StringBuilder()
    private val externalParams = mutableListOf<ExternalParameterRef>()
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

    // --- Multi-output support ------------------------------------------------
    // A node may produce several results (e.g. `split` -> N chunks). Each output
    // port gets its own SSA name; port 0 stays keyed by the bare nodeId so all
    // existing single-output callers are unchanged.
    private fun key(nodeId: String, port: Int): String = if (port == 0) nodeId else "$nodeId#$port"

    /** Set the SSA value name for a specific output port of a node. */
    public fun setValueName(nodeId: String, port: Int, valueName: String) {
        valueNames[key(nodeId, port)] = valueName
    }

    /** Get the SSA value name for a specific output port of a node. */
    public fun getValueName(nodeId: String, port: Int): String? = valueNames[key(nodeId, port)]

    /**
     * Resolve a node's input operands in input-port order, honoring the source
     * output port of each incoming edge (so a consumer of `split`'s chunk N gets
     * chunk N, not chunk 0). Equivalent to the old node-based resolution for
     * single-output producers (all source ports are 0).
     */
    public fun resolveOperands(node: GraphNode): List<String> {
        val g = graph ?: return emptyList()
        return g.edges
            .filter { it.destination.id == node.id }
            .sortedBy { it.destinationInputIndex }
            .mapNotNull { getValueName(it.source.id, it.sourceOutputIndex) }
    }

    /**
     * Record the MLIR tensor type associated with an SSA value name.
     *
     * Lets converters look up the *declared* type of an operand — the
     * type it actually has when the op consumes it — instead of having
     * to re-derive it from downstream node.inputs metadata, which can
     * reflect a post-op shape rather than the operand's true shape.
     * Seeded for `%argN` by StableHloConverter when the function
     * signature is emitted, then populated for each op's result.
     */
    public fun setValueType(valueName: String, mlirType: String) {
        valueTypes[valueName] = mlirType
    }

    /**
     * Get the MLIR tensor type for an SSA value name, or null if the
     * value was produced by a converter that did not record its type.
     */
    public fun getValueType(valueName: String): String? = valueTypes[valueName]
    
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
     * Emit a module-scope declaration (e.g. `util.global private @w : ...`).
     *
     * Module-scope lines sit between `module {` and the enclosing
     * `func.func` in the final MLIR output. [StableHloConverter]
     * buffers them separately so callers can emit them at any point
     * during node processing without disturbing the function body.
     */
    public fun emitModuleDeclaration(line: String) {
        moduleDeclarationsBuilder.appendLine("  $line")
    }

    /**
     * Return every module-scope declaration emitted so far. Used by
     * [StableHloConverter] when assembling the final content.
     */
    public fun getModuleDeclarations(): String = moduleDeclarationsBuilder.toString()

    /**
     * Register an externalized constant tensor. The converter records
     * these alongside MLIR emission so a downstream packager (see PR C
     * in issue #523) can write them into an IREE `.irpa` archive.
     */
    public fun registerExternalParameter(ref: ExternalParameterRef) {
        externalParams += ref
    }

    /**
     * Snapshot of every externalized constant registered during this
     * conversion. Surfaced on [StableHloModule.externalParameters].
     */
    public fun getExternalParameters(): List<ExternalParameterRef> = externalParams.toList()

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
        valueTypes.clear()
        stringBuilder.clear()
        moduleDeclarationsBuilder.clear()
        externalParams.clear()
        tempCounter = 0
    }
}