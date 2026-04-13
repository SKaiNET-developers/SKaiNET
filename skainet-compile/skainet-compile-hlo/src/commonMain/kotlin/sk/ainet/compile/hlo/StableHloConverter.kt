package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.tensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Main converter class that orchestrates the conversion process from ComputeGraph to StableHLO MLIR.
 * 
 * This class provides a modular architecture for converting computational graphs to StableHLO format,
 * using a registry-based system for operation mapping and a conversion context for state management.
 */
public class StableHloConverter(
    private val registry: StableHloOperationRegistry,
    private val typeMapper: TypeMapper,
    private val validator: MlirValidator? = null
) {
    
    /**
     * Convert a ComputeGraph to StableHLO MLIR format
     */
    public fun convert(graph: ComputeGraph, functionName: String = "main"): StableHloModule {
        val context = ConversionContext(typeMapper, graph)
        
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
        // single name -> encoding map. Emitting this as a structured
        // MLIR attribute on the module header lets downstream tools
        // enumerate every encoded tensor via one attribute lookup
        // instead of string-matching against scattered comments.
        val tensorEncodings = collectTensorEncodings(topo)

        // Start building MLIR content — promote to `module attributes`
        // only when we have at least one encoded tensor. Dense graphs
        // keep the bare `module {` header for byte-for-byte backward
        // compatibility with existing round-trip tests.
        if (tensorEncodings.isNotEmpty()) {
            val dictEntries = tensorEncodings.entries
                .sortedBy { it.key }
                .joinToString(", ") { (name, encoding) -> "$name = \"${encoding.name}\"" }
            context.emitLine("module attributes {skainet.tensor_encodings = {$dictEntries}} {")
        } else {
            context.emitLine("module {")
        }
        context.emitLine("  func.func $functionSignature {")
        
        // Initialize input values in context
        initializeInputValues(inputNodes, context)
        
        // Process nodes in topological order
        processNodes(topo, context)
        
        // Generate return statement with output values
        generateReturnStatement(outputNodes, context)
        
        // Close function and module
        context.emitLine("  }")
        context.emitLine("}")
        
        val content = context.getContent()
        
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
            outputSpecs = outputSpecs
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
                context.emitComment("Error processing node ${node.id}: ${e.message}")
                context.emitComment("Unsupported op ${node.operation.name} (type=${node.operation.type}) for node ${node.id}")
            }
        }
    }
    
    private fun processNode(node: GraphNode, context: ConversionContext) {
        val converter = registry.getConverter(node.operation.name)
            ?: throw UnsupportedOperationException("No converter found for operation: ${node.operation.name}")

        // Get input operands from context
        val inputNodes = context.getInputNodes(node)
        val operands = inputNodes.mapNotNull { context.getValueName(it.id) }

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
            }
            is ConversionResult.Failure -> {
                context.emitComment("Conversion failed for node ${node.id}: ${result.error}")
                result.fallbackComment?.let { context.emitComment(it) }
            }
            is ConversionResult.Unsupported -> {
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
    private fun collectTensorEncodings(nodes: List<GraphNode>): Map<String, TensorEncoding> {
        val result = linkedMapOf<String, TensorEncoding>()
        for (node in nodes) {
            for (spec in node.outputs) {
                val encoding = spec.tensorEncoding ?: continue
                if (spec.name !in result) result[spec.name] = encoding
            }
            for (spec in node.inputs) {
                val encoding = spec.tensorEncoding ?: continue
                if (spec.name !in result) result[spec.name] = encoding
            }
        }
        return result
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