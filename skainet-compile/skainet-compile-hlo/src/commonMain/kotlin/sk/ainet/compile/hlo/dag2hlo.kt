package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Represents a StableHLO MLIR module text output
 */
public data class StableHloModule(
    val content: String,
    val functionName: String = "main",
    val inputSpecs: List<TensorSpec> = emptyList(),
    val outputSpecs: List<TensorSpec> = emptyList(),
    val metadata: Map<String, Any> = emptyMap(),
    /**
     * Constants that were lifted out of [content] as `util.global`
     * references under [ConstantMaterializationPolicy.ExternalAlways].
     * A downstream packager writes these into an IREE `.irpa`; see
     * issue #523. Empty under the default inline policy.
     */
    val externalParameters: List<ExternalParameterRef> = emptyList()
) {
    /**
     * Validate this module using the provided validator
     */
    public fun validate(validator: MlirValidator): List<String> {
        return validator.validate(content)
    }
    
    /**
     * Optimize this module using the provided optimizer
     */
    public fun optimize(optimizer: StableHloOptimizer): StableHloModule {
        return optimizer.optimize(this)
    }

    /**
     * Lazily iterate over the MLIR content line by line.
     */
    public fun contentLines(): Sequence<String> = content.lineSequence()
}

/**
 * Export a ComputeGraph into a minimal StableHLO MLIR module text.
 *
 * Notes:
 * - This function now uses the new modular converter architecture for better extensibility.
 * - It maintains backward compatibility with the original implementation.
 * - Currently supports: input, add, matmul, relu. Unsupported ops are emitted as comments.
 * - DType mapping expects TensorSpec.dtype to be strings like "FP32", "F32", "F64", "I32".
 */
public fun toStableHlo(
    graph: ComputeGraph,
    functionName: String = "main",
    /**
     * Selected compile target (iree device id, e.g. "torq", "llvm-cpu"); `null` =
     * target-agnostic emission (the portable default, unchanged behavior).
     */
    target: String? = null,
    /**
     * Per-target op-granularity policy (fused vs decomposed). Resolve it at the call
     * site with `TargetOptimizers.granularityFor(target)` and pass it in, so the emitter
     * stays decoupled from the optimizer registry. `null` = decompose everything.
     */
    granularity: sk.ainet.compile.target.OpGranularityPolicy? = null,
): StableHloModule {
    // Use the new converter architecture
    val converter = StableHloConverterFactory.createBasic(target = target, granularity = granularity)
    return converter.convert(graph, functionName)
}

/**
 * Export a [sk.ainet.lang.graph.ResolvedComputeGraph] into a StableHLO
 * MLIR module — the dtype-resolved entry point that the W7
 * `DTypeConstraintResolutionPass` produces.
 *
 * The contract this overload upholds vs the plain [ComputeGraph]
 * variant: every edge's dtype has already been resolved to a typed
 * [sk.ainet.lang.types.DType] (the wrapper's `validate()` would
 * have caught any unparseable strings), and every node carries the
 * `dtype_resolved` marker from the pass. Callers that flow through
 * this overload get a precondition guarantee that the HLO emit
 * step won't silently misinterpret a stray dtype string.
 *
 * Today the converter still consumes the underlying [ComputeGraph] —
 * the wrapper is the *contract*, not a separate emit path. As
 * future passes start writing layout / backend metadata into the
 * resolved graph, the converter can read those typed accessors
 * directly. This entry point gives them the stable hook to do so.
 */
public fun toStableHlo(
    graph: sk.ainet.lang.graph.ResolvedComputeGraph,
    functionName: String = "main",
    validate: Boolean = true,
    target: String? = null,
    granularity: sk.ainet.compile.target.OpGranularityPolicy? = null,
): StableHloModule {
    if (validate) {
        graph.validate().requireValid()
    }
    // Delegate to the underlying ComputeGraph emit path — same HLO output
    // for graphs that pass validation. Future versions can branch here to
    // consume `graph.resolvedLayout(edgeId)` / `graph.backendAssignment(nodeId)`
    // once those passes ship.
    return toStableHlo(graph.delegate, functionName, target, granularity)
}

/**
 * Legacy implementation for backward compatibility.
 * 
 * @deprecated Use StableHloConverter directly for better control and extensibility.
 */
@Deprecated("Use StableHloConverter directly", ReplaceWith("StableHloConverterFactory.createBasic().convert(graph, functionName)"))
public fun toStableHloLegacy(graph: ComputeGraph, functionName: String = "main"): StableHloModule {
    val topo = graph.getTopologicalOrder()
    val sb = StringBuilder()

    // Collect inputs as nodes with type "input"
    val inputNodes = topo.filter { it.operation.type == "input" || it.operation.name == "input" }

    fun mlirElemType(dtype: String): String = when (dtype.uppercase()) {
        "FP32", "F32" -> "f32"
        "FP64", "F64" -> "f64"
        "I32" -> "i32"
        "I64" -> "i64"
        else -> "f32" // default fallback
    }

    fun mlirShape(spec: TensorSpec): String {
        val shapeStr = spec.shape?.joinToString(",") ?: "?"
        return "tensor<${shapeStr.ifEmpty { "?" }}x${mlirElemType(spec.dtype)}>"
    }

    // Build function signature from input nodes' first output spec (or metadata)
    val argsSig = inputNodes.mapIndexed { idx, node ->
        val outSpec = node.outputs.firstOrNull() ?: TensorSpec("arg$idx", emptyList(), "FP32")
        "%arg$idx: ${mlirShape(outSpec)}"
    }.joinToString(", ")

    sb.appendLine("module {")
    sb.appendLine("  func.func @${functionName}(${argsSig}) -> () {")

    // Map from node id to MLIR SSA value name
    val valueNames = mutableMapOf<String, String>()

    // Seed inputs
    inputNodes.forEachIndexed { idx, node ->
        valueNames[node.id] = "%arg$idx"
        // If the input node has a friendly name in outputs, annotate
        node.outputs.firstOrNull()?.let { spec ->
            sb.appendLine("    // input ${node.id}: ${spec.name} : ${mlirShape(spec)}")
        }
    }

    // Emit operations in topological order
    var tmpCounter = 0
    fun nextTmp(): String = "%v${tmpCounter++}"

    topo.forEach { node ->
        // Skip inputs, already mapped
        if (node.operation.type == "input" || node.operation.name == "input") return@forEach

        // Resolve operand SSA names from input nodes connected in graph
        val inputs = graph.getInputNodes(node)
        val operandValues = inputs.mapNotNull { valueNames[it.id] }

        // Determine output spec/type for printing
        val outSpec = node.outputs.firstOrNull()

        when (node.operation.name.lowercase()) {
            "add" -> {
                if (operandValues.size == 2) {
                    val res = nextTmp()
                    val ty = outSpec?.let { mlirShape(it) } ?: "tensor<?xf32>"
                    sb.appendLine("    $res = stablehlo.add ${operandValues[0]}, ${operandValues[1]} : $ty")
                    valueNames[node.id] = res
                } else {
                    sb.appendLine("    // Unsupported add arity for node ${node.id}")
                }
            }
            "matmul" -> {
                if (operandValues.size == 2) {
                    val res = nextTmp()
                    val ty = outSpec?.let { mlirShape(it) } ?: "tensor<?x?xf32>"
                    // Minimal dot_general with default contracting dimensions for last dim
                    sb.appendLine("    $res = stablehlo.dot_general ${operandValues[0]}, ${operandValues[1]} ")
                    sb.appendLine("      contracting_dims = [[-1], [-2]] : $ty")
                    valueNames[node.id] = res
                } else {
                    sb.appendLine("    // Unsupported matmul arity for node ${node.id}")
                }
            }
            "relu" -> {
                if (operandValues.size == 1) {
                    val res = nextTmp()
                    val ty = outSpec?.let { mlirShape(it) } ?: "tensor<?xf32>"
                    val zeroConst = nextTmp()
                    val elem = outSpec?.let { mlirElemType(it.dtype) } ?: "f32"
                    sb.appendLine("    $zeroConst = stablehlo.constant dense<0.0> : $ty")
                    sb.appendLine("    $res = stablehlo.maximum ${operandValues[0]}, $zeroConst : $ty")
                    valueNames[node.id] = res
                } else {
                    sb.appendLine("    // Unsupported relu arity for node ${node.id}")
                }
            }
            else -> {
                sb.appendLine("    // Unsupported op ${node.operation.name} (type=${node.operation.type}) for node ${node.id}")
            }
        }
    }

    // For now, no explicit return values
    sb.appendLine("    return")
    sb.appendLine("  }")
    sb.appendLine("}")

    return StableHloModule(sb.toString())
}