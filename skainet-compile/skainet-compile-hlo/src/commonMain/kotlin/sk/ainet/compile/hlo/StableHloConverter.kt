package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.blockOrder
import sk.ainet.lang.tensor.ops.tensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Main converter class that orchestrates the conversion process from ComputeGraph to StableHLO MLIR.
 * 
 * This class provides a modular architecture for converting computational graphs to StableHLO format,
 * using a registry-based system for operation mapping and a conversion context for state management.
 */
public class StableHloConverter @kotlin.jvm.JvmOverloads constructor(
    private val registry: StableHloOperationRegistry,
    private val typeMapper: TypeMapper,
    private val validator: MlirValidator? = null,
    /**
     * Governs how constant tensors are materialized into the emitted
     * MLIR — inline `dense<...>` (default, historical) or lifted out
     * behind `util.global.load` references (see issue #523). Handed
     * through to every [ConversionContext] this converter creates.
     */
    private val materializationPolicy: ConstantMaterializationPolicy =
        ConstantMaterializationPolicy.InlineAlways,
    /** Selected compile target (iree device id, e.g. "torq"); handed to every context. */
    private val target: String? = null,
    /** Per-target op-granularity policy; handed to every context (null = decompose all). */
    private val granularity: sk.ainet.compile.target.OpGranularityPolicy? = null,
    /**
     * How conversion failures behave: [ConversionErrorPolicy.STRICT] (default)
     * throws on the first node that fails to lower; [ConversionErrorPolicy.LENIENT]
     * restores the historical comment-and-continue behavior (issue #1247).
     */
    private val errorPolicy: ConversionErrorPolicy = ConversionErrorPolicy.STRICT
) {

    /**
     * Convert a ComputeGraph to StableHLO MLIR format
     */
    public fun convert(graph: ComputeGraph, functionName: String = "main"): StableHloModule {
        val context = ConversionContext(typeMapper, graph, materializationPolicy, target, granularity, errorPolicy)
        
        // Pre-conversion validation (allow orphaned nodes for backward compatibility)
        val validationResult = graph.validate()
        if (validationResult is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            // Check if the only errors are orphaned nodes - if so, proceed anyway for backward compatibility
            val nonOrphanedErrors = validationResult.errors.filter { !it.contains("Orphaned nodes found") }
            if (nonOrphanedErrors.isNotEmpty()) {
                throw IllegalArgumentException("Invalid graph: $nonOrphanedErrors")
            }
            // If only orphaned node errors, log a warning but continue
        }
        
        // Get topological order for processing
        val topo = graph.getTopologicalOrder()
        
        // Collect input and output nodes
        val inputNodes = topo.filter { it.operation.type == "input" || it.operation.name == "input" }
        val outputNodes = graph.getOutputNodes()
        
        // Determine output specifications from output nodes
        val outputSpecs = determineOutputSpecs(outputNodes)
        
        // Build function signature with proper return types
        val functionSignature = buildFunctionSignature(inputNodes, outputSpecs, functionName)

        // Collect every TensorSpec with a non-null tensorEncoding into a
        // single name -> structural-facts map. Emitting this as a structured
        // MLIR attribute on the module header lets downstream tools
        // enumerate every encoded tensor via one attribute lookup —
        // block sizes and bit widths as integers, not display names (#1179).
        val structuralLayouts = collectStructuralLayouts(topo)

        // Process nodes first, then assemble the final content.
        // Converters populate two buffers on the context — op emissions
        // for the function body and any module-scope declarations
        // (e.g. `util.global` decls under an external-materialization
        // policy). Deferring assembly lets us inject module-scope
        // lines between `module {` and `func.func` without string
        // surgery.
        initializeInputValues(inputNodes, context)
        processNodes(topo, context)
        generateReturnStatement(outputNodes, context)

        val headerAttributes = mutableListOf<String>()
        if (structuralLayouts.isNotEmpty()) {
            val layoutEntries = structuralLayouts.entries
                .sortedBy { it.key }
                .joinToString(", ") { (name, attr) -> "$name = $attr" }
            headerAttributes += "skainet.tensor_layouts = {$layoutEntries}"
        }
        // SKEEP-005: schedule hints ride the module header next to the layouts — one graph, extra
        // schedule metadata. Keyed by node id; a consumer that ignores it computes the same result.
        val schedules = collectScheduleHints(topo)
        if (schedules.isNotEmpty()) {
            headerAttributes += "skainet.schedule = {" + schedules.joinToString(", ") { (id, hint) -> "${mlirKey(id)} = ${scheduleAttr(hint)}" } + "}"
        }
        val moduleHeader = if (headerAttributes.isEmpty()) "module {" else "module attributes {${headerAttributes.joinToString(", ")}} {"

        val assembled = StringBuilder()
        assembled.appendLine(moduleHeader)
        val moduleDecls = context.getModuleDeclarations()
        if (moduleDecls.isNotEmpty()) {
            assembled.append(moduleDecls)
        }
        assembled.appendLine("  func.func $functionSignature {")
        assembled.append(context.getContent())
        assembled.appendLine("  }")
        assembled.appendLine("}")

        val content = assembled.toString()
        
        // Optional validation of generated MLIR
        validator?.validate(content)?.let { errors ->
            if (errors.isNotEmpty()) {
                throw IllegalStateException("Generated MLIR validation failed: $errors")
            }
        }
        
        return StableHloModule(
            content = content,
            functionName = functionName,
            inputSpecs = inputNodes.mapNotNull { it.outputs.firstOrNull() },
            outputSpecs = outputSpecs,
            externalParameters = context.getExternalParameters()
        )
    }
    
    /**
     * Convert with optimization passes applied
     */
    public fun convertWithOptimization(graph: ComputeGraph, optimizer: StableHloOptimizer): StableHloModule {
        val module = convert(graph)
        return optimizer.optimize(module)
    }
    
    private fun buildFunctionSignature(inputNodes: List<GraphNode>, outputSpecs: List<TensorSpec>, functionName: String): String {
        val argsSig = inputNodes.mapIndexed { idx, node ->
            val outSpec = node.outputs.firstOrNull() ?: TensorSpec("arg$idx", emptyList(), "FP32")
            "%arg$idx: ${typeMapper.mapTensorType(outSpec)}"
        }.joinToString(", ")
        
        val returnSig = if (outputSpecs.isNotEmpty()) {
            outputSpecs.joinToString(", ") { typeMapper.mapTensorType(it) }
        } else {
            ""
        }
        
        return if (returnSig.isNotEmpty()) {
            "@${functionName}(${argsSig}) -> (${returnSig})"
        } else {
            "@${functionName}(${argsSig}) -> ()"
        }
    }
    
    private fun initializeInputValues(inputNodes: List<GraphNode>, context: ConversionContext) {
        inputNodes.forEachIndexed { idx, node ->
            val valueName = "%arg$idx"
            context.setValueName(node.id, valueName)

            // Seed the SSA type map with %argN's declared function-signature
            // type so downstream ops can recover the operand type via
            // context.getValueType(operands[0]) instead of re-deriving it
            // (see issue #518).
            node.outputs.firstOrNull()?.let { spec ->
                context.setValueType(valueName, typeMapper.mapTensorType(spec))
            }

            // Add comment for clarity
            node.outputs.firstOrNull()?.let { spec ->
                context.emitComment("input ${node.id}: ${spec.name} : ${typeMapper.mapTensorType(spec)}")
            }

            // Preserve any physical storage encoding carried on the input
            // spec (Q4_K / Q8_0 / TernaryPacked / TurboQuant / …) as an
            // MLIR comment so downstream tools see that quantization
            // flowed through. No-op when the spec has no encoding.
            node.outputs.forEachIndexed { outIdx, spec ->
                context.emitEncodingAnnotation(role = "input", index = outIdx, spec = spec)
            }
        }
    }
    
    private fun processNodes(nodes: List<GraphNode>, context: ConversionContext) {
        for (node in nodes) {
            // Skip input nodes as they're already processed
            if (node.operation.type == "input" || node.operation.name == "input") {
                continue
            }
            
            try {
                processNode(node, context)
            } catch (e: Exception) {
                if (errorPolicy == ConversionErrorPolicy.STRICT) {
                    // Already-precise diagnostics pass through unwrapped.
                    if (e is HloConversionException || e is MissingOperandException) throw e
                    // Quote the name so trailing whitespace / casing surprises are
                    // visible, and include the registry's full key set so "no
                    // converter found" failures are self-diagnostic. Note the cause
                    // may be an unrelated throw from a registered converter — a
                    // known name here does not mean a registry miss (issue #1247).
                    val known = registry.getSupportedOperations().sorted().joinToString(", ")
                    throw HloConversionException(
                        "Error processing node ${node.id}: op '${node.operation.name}' " +
                            "(type=${node.operation.type}) threw ${e::class.simpleName}: " +
                            "${e.message}. Registry known names: [$known]",
                        e
                    )
                }
                context.emitComment("Error processing node ${node.id}: ${e.message}")
                // Quote the name so trailing whitespace / casing surprises are visible,
                // and include the registry's full key set so "no converter found"
                // failures are self-diagnostic (is the name missing, or mis-matched?).
                val known = registry.getSupportedOperations().sorted().joinToString(", ")
                context.emitComment(
                    "Unsupported op '${node.operation.name}' (type=${node.operation.type}) " +
                        "for node ${node.id}. Known names: [$known]"
                )
            }
        }
    }
    
    private fun processNode(node: GraphNode, context: ConversionContext) {
        val converter = registry.getConverter(node.operation.name)
            ?: throw UnsupportedOperationException("No converter found for operation: ${node.operation.name}")

        // Get input operands in input-port order, honoring each incoming edge's
        // source output port so consumers of a multi-output op (e.g. split) get
        // the right chunk. Equivalent to the prior node-based resolution for
        // single-output producers.
        val operands = context.resolveOperands(node)

        // Surface any physical storage encoding declared on this node's
        // result specs as an MLIR comment before the operation is
        // emitted. Converters that want finer-grained placement can call
        // ConversionContext.emitEncodingAnnotation themselves.
        node.outputs.forEachIndexed { outIdx, spec ->
            context.emitEncodingAnnotation(role = "result", index = outIdx, spec = spec)
        }

        // Convert the operation
        val result = converter.convert(node, operands, context)
        
        when (result) {
            is ConversionResult.Success -> {
                context.setValueName(node.id, result.outputValueName)
                // Record the result's MLIR type so downstream operands can
                // look it up (see issue #518). Uses the node's declared
                // first output spec — converters that produce types
                // differing from node.outputs[0] can override this by
                // calling context.setValueType directly.
                node.outputs.firstOrNull()?.let { spec ->
                    context.setValueType(result.outputValueName, typeMapper.mapTensorType(spec))
                }
            }
            is ConversionResult.Failure -> {
                if (errorPolicy == ConversionErrorPolicy.STRICT) {
                    throw HloConversionException(
                        "Conversion failed for node ${node.id} (op '${node.operation.name}'): ${result.error}"
                    )
                }
                context.emitComment("Conversion failed for node ${node.id}: ${result.error}")
                result.fallbackComment?.let { context.emitComment(it) }
            }
            is ConversionResult.Unsupported -> {
                if (errorPolicy == ConversionErrorPolicy.STRICT) {
                    throw HloConversionException(
                        "Unsupported operation ${result.operationName} for node ${node.id}: ${result.reason}"
                    )
                }
                context.emitComment("Unsupported operation ${result.operationName}: ${result.reason}")
            }
        }
    }
    
    /**
     * Walk every node's input and output specs once and collect the
     * `name -> encoding` map of every tensor that carries a non-null
     * [TensorEncoding]. Duplicates (the same name appearing in multiple
     * nodes) collapse to a single entry — first-writer-wins.
     *
     * Implementation note: uses an explicit `!in` check rather than
     * `MutableMap.putIfAbsent` because the latter is a JVM-only
     * extension and does not exist in Kotlin common-stdlib for WasmJS
     * / JS / Native targets.
     */
    /**
     * Structural per-tensor storage facts for the module header (#1179): unlike
     * `skainet.tensor_encodings` (display names, kept for compatibility), these entries are
     * machine-readable — block element counts, block bytes, bit widths, block order — so a
     * downstream consumer can size and address packed weights without a lookup table of names.
     */
    /** `(nodeId, hint)` for every node carrying a schedule hint, in topological order. */
    private fun collectScheduleHints(topo: List<sk.ainet.lang.graph.GraphNode>): List<Pair<String, sk.ainet.context.schedule.ScheduleHint>> =
        topo.mapNotNull { node ->
            val hint = sk.ainet.context.schedule.ScheduleHint.fromAttribute(node.metadata[sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY])
                ?: sk.ainet.context.schedule.ScheduleHint.fromAttribute(node.operation.parameters[sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY])
            hint?.let { node.id to it }
        }

    private fun scheduleAttr(hint: sk.ainet.context.schedule.ScheduleHint): String {
        val dims = hint.parallelDims.joinToString(", ") { "\"$it\"" }
        val p = hint.parallelism?.let { ", parallelism = $it" } ?: ""
        return "{parallel_dims = [$dims]$p}"
    }

    /** MLIR dictionary keys must be bare identifiers or quoted strings. */
    private fun mlirKey(id: String): String =
        if (id.isNotEmpty() && (id[0].isLetter() || id[0] == '_') && id.all { it.isLetterOrDigit() || it == '_' || it == '$' || it == '.' }) id else "\"$id\""

    private fun collectStructuralLayouts(nodes: List<GraphNode>): Map<String, String> {
        val result = linkedMapOf<String, String>()
        for (node in nodes) {
            for (spec in node.outputs + node.inputs) {
                val encoding = spec.tensorEncoding ?: continue
                if (spec.name in result) continue
                result[spec.name] = structuralAttr(encoding, spec.blockOrder)
            }
        }
        return result
    }

    private fun structuralAttr(encoding: TensorEncoding, blockOrder: String?): String {
        val facts = buildList {
            add("kind = \"" + encoding.name + "\"")
            when (encoding) {
                is TensorEncoding.Dense -> add("bytes_per_element = " + encoding.bytesPerElement)
                TensorEncoding.Q4_K -> { add("block_elems = " + TensorEncoding.Q4_K.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q4_K.BYTES_PER_BLOCK) }
                TensorEncoding.Q5_K -> { add("block_elems = " + TensorEncoding.Q5_K.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q5_K.BYTES_PER_BLOCK) }
                TensorEncoding.Q6_K -> { add("block_elems = " + TensorEncoding.Q6_K.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q6_K.BYTES_PER_BLOCK) }
                TensorEncoding.Q4_0 -> { add("block_elems = " + TensorEncoding.Q4_0.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q4_0.BYTES_PER_BLOCK) }
                TensorEncoding.Q8_0 -> { add("block_elems = " + TensorEncoding.Q8_0.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q8_0.BYTES_PER_BLOCK) }
                TensorEncoding.Q5_0 -> { add("block_elems = " + TensorEncoding.Q5_0.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q5_0.BYTES_PER_BLOCK) }
                TensorEncoding.Q5_1 -> { add("block_elems = " + TensorEncoding.Q5_1.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.Q5_1.BYTES_PER_BLOCK) }
                TensorEncoding.TQ1_0 -> { add("block_elems = " + TensorEncoding.TQ1_0.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.TQ1_0.BYTES_PER_BLOCK) }
                TensorEncoding.TQ2_0 -> { add("block_elems = " + TensorEncoding.TQ2_0.BLOCK_SIZE); add("block_bytes = " + TensorEncoding.TQ2_0.BYTES_PER_BLOCK) }
                is TensorEncoding.TurboQuantPolar -> { add("bits = " + encoding.bitsPerElement); add("block_elems = " + encoding.blockSize) }
                TensorEncoding.TernaryPacked -> add("bits = 2")
                else -> { /* kind alone: unknown structure is stated, not invented */ }
            }
            if (blockOrder != null) add("block_order = \"" + blockOrder + "\"")
        }
        return "{" + facts.joinToString(", ") + "}"
    }

    /**
     * Determine output specifications from output nodes
     */
    private fun determineOutputSpecs(outputNodes: List<GraphNode>): List<TensorSpec> {
        return outputNodes.mapNotNull { node ->
            // For output nodes, we want their output specifications
            // If a node has multiple outputs, we take the first one
            // In the future, this could be enhanced to handle multiple outputs per node
            node.outputs.firstOrNull()
        }
    }
    
    /**
     * Generate the return statement with output values
     */
    private fun generateReturnStatement(outputNodes: List<GraphNode>, context: ConversionContext) {
        if (outputNodes.isEmpty()) {
            // No outputs - just return
            context.emitLine("    return")
        } else {
            // Get the SSA value names for output nodes
            val outputValues = outputNodes.mapNotNull { node ->
                context.getValueName(node.id)
            }
            
            if (outputValues.isEmpty()) {
                // No output values found - this might happen if output nodes failed to convert
                if (errorPolicy == ConversionErrorPolicy.STRICT) {
                    throw HloConversionException(
                        "Module produced no return values: none of the ${outputNodes.size} " +
                            "output node(s) [${outputNodes.joinToString(", ") { it.id }}] was " +
                            "successfully converted. A compute-free module with an empty " +
                            "return is never servable (issue #1247)."
                    )
                }
                context.emitComment("Warning: No output values found for return statement")
                context.emitLine("    return")
            } else {
                // Return the output values
                val returnValues = outputValues.joinToString(", ")
                context.emitLine("    return $returnValues : ${buildReturnTypeSignature(outputNodes)}")
            }
        }
    }
    
    /**
     * Build the return type signature for the return statement
     */
    private fun buildReturnTypeSignature(outputNodes: List<GraphNode>): String {
        val outputSpecs = outputNodes.mapNotNull { it.outputs.firstOrNull() }
        return if (outputSpecs.isNotEmpty()) {
            outputSpecs.joinToString(", ") { typeMapper.mapTensorType(it) }
        } else {
            ""
        }
    }
}