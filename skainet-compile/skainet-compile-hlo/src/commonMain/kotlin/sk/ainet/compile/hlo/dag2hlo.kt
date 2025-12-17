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
    val metadata: Map<String, Any> = emptyMap()
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
public fun toStableHlo(graph: ComputeGraph, functionName: String = "main"): StableHloModule {
    // Use the new converter architecture
    val converter = StableHloConverterFactory.createBasic()
    return converter.convert(graph, functionName)
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