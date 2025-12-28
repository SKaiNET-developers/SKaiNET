package sk.ainet.compile.c

import sk.ainet.compile.c.templates.HeaderTemplate
import sk.ainet.compile.c.templates.SourceTemplate
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph

/**
 * Facade to export a high-level model or a prebuilt ComputeGraph into Arduino C code.
 * 
 * This facade follows the ModelExportFacade pattern from skainet-compile-json,
 * providing consistent API patterns for different input types while integrating
 * CCodeGenerator and ArduinoLibraryPackager components.
 * 
 * Capabilities:
 * - If [model] is already a ComputeGraph, delegates to [exportGraphToArduinoLibrary].
 * - Otherwise, prefer using the overload that accepts a [forwardPass] lambda to run a single
 *   forward execution under a recording tape; the tape is then converted to a ComputeGraph and exported.
 */
public class CCodegenFacade {
    
    /**
     * Exports a model to an Arduino library.
     * 
     * @param model The model to export (preferably a ComputeGraph)
     * @param outputPath Base path where the library directory will be created
     * @param libraryName Name of the Arduino library (defaults to model class name)
     * @return ArduinoLibraryResult containing information about the generated library
     */
    public fun <T : Any> exportToArduinoLibrary(
        model: T,
        outputPath: String,
        libraryName: String = model::class.simpleName ?: "SKaiNETModel"
    ): ArduinoLibraryResult {
        return when (model) {
            is ComputeGraph -> exportGraphToArduinoLibrary(model, outputPath, libraryName)
            else -> {
                error(
                    buildString {
                        // Avoid KClass.qualifiedName which is not supported on Kotlin/JS
                        val typeName = model::class.simpleName ?: "unknown"
                        appendLine("exportToArduinoLibrary(model) does not have a direct adapter for type: $typeName.")
                        appendLine("If you have a ComputeGraph already, call exportGraphToArduinoLibrary(graph, outputPath, libraryName) instead.")
                        appendLine(
                            "Otherwise, call exportToArduinoLibrary(model, forwardPass = { /* run a single forward pass here */ }, outputPath, libraryName) " +
                                "so the facade can record execution to a tape, build a ComputeGraph, and export Arduino C code."
                        )
                    }
                )
            }
        }
    }
    
    /**
     * Fallback overload for models without a direct adapter.
     * 
     * Provide a [forwardPass] that executes exactly one forward run of the model with example inputs.
     * The facade will:
     * 1) Start a recording tape
     * 2) Execute [forwardPass]
     * 3) Convert the tape to a [ComputeGraph]
     * 4) Delegate to [exportGraphToArduinoLibrary]
     * 
     * @param model The model to export
     * @param forwardPass Lambda that executes one forward pass of the model
     * @param outputPath Base path where the library directory will be created
     * @param libraryName Name of the Arduino library (defaults to model class name)
     * @return ArduinoLibraryResult containing information about the generated library
     */
    public fun <T : Any> exportToArduinoLibrary(
        model: T,
        forwardPass: () -> Unit,
        outputPath: String,
        libraryName: String = model::class.simpleName ?: "SKaiNETModel"
    ): ArduinoLibraryResult {
        // If caller passed a graph as model, keep the fast path
        if (model is ComputeGraph) return exportGraphToArduinoLibrary(model, outputPath, libraryName)

        // Record a single forward pass under the new graph/tape execution context.
        // Bridge this context's tape to the global Execution.tapeStack so callers using
        // Execution.recordingOps(baseOps) will record into the same tape.
        // Use a functional baseOps for tracing so relu/sigmoid/etc. execute and produce traces
        val ctx = DefaultGraphExecutionContext.tape(baseOps = sk.ainet.lang.tensor.ops.VoidTensorOps())
        val (tape, _) = ctx.record {
            // Push the current tape into the global stack for the duration of forwardPass
            val globalStack = sk.ainet.tape.Execution.tapeStack
            val pushed = ctx.currentTape
            if (pushed != null) {
                globalStack.pushTape(pushed)
            }
            try {
                forwardPass()
            } finally {
                if (pushed != null) {
                    globalStack.popTape()
                }
            }
        }
        
        // Prefer DefaultExecutionTape.toComputeGraph() which builds a real graph from traces/ops.
        val graph = when (tape) {
            is DefaultExecutionTape -> tape.toComputeGraph()
            // Fallback to the generic extension (may be a stub in some builds)
            else -> tape?.toComputeGraph() ?: sk.ainet.lang.graph.DefaultComputeGraph()
        }
        
        return exportGraphToArduinoLibrary(graph, outputPath, libraryName)
    }
    
    /**
     * Exports a ComputeGraph directly to an Arduino library.
     * 
     * This method integrates CCodeGenerator and ArduinoLibraryPackager to create
     * a complete Arduino library from a validated ComputeGraph.
     * 
     * @param graph The ComputeGraph to export
     * @param outputPath Base path where the library directory will be created
     * @param libraryName Name of the Arduino library
     * @return ArduinoLibraryResult containing information about the generated library
     */
    public fun exportGraphToArduinoLibrary(
        graph: ComputeGraph,
        outputPath: String,
        libraryName: String
    ): ArduinoLibraryResult {
        require(outputPath.isNotBlank()) { "outputPath cannot be blank" }
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        
        // Create and validate the C code generator
        val codeGenerator = CCodeGenerator(graph)
        val validation = codeGenerator.validateGraph()
        if (validation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            throw IllegalArgumentException("Graph validation failed: ${validation.errors.joinToString("; ")}")
        }
        
        // Generate all components needed for the Arduino library
        val memoryLayout = codeGenerator.calculateMemoryRequirements()
        val layers = codeGenerator.generateAllLayers()
        val weights = codeGenerator.extractWeights()
        
        // Determine input/output dimensions from the graph
        val inputDims = determineInputDimensions(graph)
        val outputDims = determineOutputDimensions(graph)
        
        // Generate C code using templates
        val headerCode = HeaderTemplate.generate(
            libraryName = libraryName,
            inputDims = inputDims,
            outputDims = outputDims,
            memoryRequirements = memoryLayout
        )
        
        val sourceCode = SourceTemplate.generate(
            libraryName = libraryName,
            layers = layers,
            weights = weights
        )
        
        // Validate generated code for static allocation and buffer alternation
        val staticValidation = codeGenerator.validateGeneratedCodeStaticAllocation(sourceCode)
        if (staticValidation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            throw IllegalArgumentException("Generated code static allocation validation failed: ${staticValidation.errors.joinToString("; ")}")
        }
        
        val bufferValidation = codeGenerator.validateGeneratedCodeBufferAlternation(sourceCode)
        if (bufferValidation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            throw IllegalArgumentException("Generated code buffer alternation validation failed: ${bufferValidation.errors.joinToString("; ")}")
        }
        
        // Package into Arduino library structure
        val packager = ArduinoLibraryPackager()
        return packager.createLibraryStructure(
            outputPath = outputPath,
            libraryName = libraryName,
            sourceCode = sourceCode,
            headerCode = headerCode,
            memoryLayout = memoryLayout,
            inputDims = inputDims,
            outputDims = outputDims
        )
    }
    
    /**
     * Determines input dimensions from the ComputeGraph.
     * 
     * @param graph The ComputeGraph to analyze
     * @return Array of input tensor dimensions
     */
    private fun determineInputDimensions(graph: ComputeGraph): IntArray {
        // Find input nodes (nodes with no predecessors)
        val inputNodes = graph.nodes.filter { node ->
            graph.getIncomingEdges(node.id).isEmpty()
        }
        
        if (inputNodes.isEmpty()) {
            throw IllegalArgumentException("Graph has no input nodes")
        }
        
        // Use the first input node's output shape as the input dimensions
        val inputNode = inputNodes.first()
        val inputShape = inputNode.outputs.firstOrNull()?.shape
            ?: throw IllegalArgumentException("Input node has no output shape")
        
        return inputShape.toIntArray()
    }
    
    /**
     * Determines output dimensions from the ComputeGraph.
     * 
     * @param graph The ComputeGraph to analyze
     * @return Array of output tensor dimensions
     */
    private fun determineOutputDimensions(graph: ComputeGraph): IntArray {
        // Find output nodes (nodes with no successors)
        val outputNodes = graph.nodes.filter { node ->
            graph.getOutgoingEdges(node.id).isEmpty()
        }
        
        if (outputNodes.isEmpty()) {
            throw IllegalArgumentException("Graph has no output nodes")
        }
        
        // Use the first output node's output shape as the output dimensions
        val outputNode = outputNodes.first()
        val outputShape = outputNode.outputs.firstOrNull()?.shape
            ?: throw IllegalArgumentException("Output node has no output shape")
        
        return outputShape.toIntArray()
    }
}