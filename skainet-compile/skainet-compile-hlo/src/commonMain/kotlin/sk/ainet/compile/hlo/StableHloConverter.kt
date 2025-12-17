package sk.ainet.compile.hlo

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec

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
        
        // Pre-conversion validation
        val validationResult = graph.validate()
        if (validationResult is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            throw IllegalArgumentException("Invalid graph: ${validationResult.errors}")
        }
        
        // Get topological order for processing
        val topo = graph.getTopologicalOrder()
        
        // Collect input nodes
        val inputNodes = topo.filter { it.operation.type == "input" || it.operation.name == "input" }
        
        // Build function signature
        val functionSignature = buildFunctionSignature(inputNodes, functionName)
        
        // Start building MLIR content
        context.emitLine("module {")
        context.emitLine("  func.func $functionSignature {")
        
        // Initialize input values in context
        initializeInputValues(inputNodes, context)
        
        // Process nodes in topological order
        processNodes(topo, context)
        
        // Close function and module
        context.emitLine("    return")
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
            outputSpecs = emptyList() // TODO: Determine output specs
        )
    }
    
    /**
     * Convert with optimization passes applied
     */
    public fun convertWithOptimization(graph: ComputeGraph, optimizer: StableHloOptimizer): StableHloModule {
        val module = convert(graph)
        return optimizer.optimize(module)
    }
    
    private fun buildFunctionSignature(inputNodes: List<GraphNode>, functionName: String): String {
        val argsSig = inputNodes.mapIndexed { idx, node ->
            val outSpec = node.outputs.firstOrNull() ?: TensorSpec("arg$idx", emptyList(), "FP32")
            "%arg$idx: ${typeMapper.mapTensorType(outSpec)}"
        }.joinToString(", ")
        
        return "@${functionName}(${argsSig}) -> ()"
    }
    
    private fun initializeInputValues(inputNodes: List<GraphNode>, context: ConversionContext) {
        inputNodes.forEachIndexed { idx, node ->
            val valueName = "%arg$idx"
            context.setValueName(node.id, valueName)
            
            // Add comment for clarity
            node.outputs.firstOrNull()?.let { spec ->
                context.emitComment("input ${node.id}: ${spec.name} : ${typeMapper.mapTensorType(spec)}")
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
}