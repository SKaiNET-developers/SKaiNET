package sk.ainet.compile.c

import sk.ainet.compile.c.templates.HeaderTemplate
import sk.ainet.compile.c.templates.SourceTemplate
import sk.ainet.context.ExecutionContext
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
        forwardPass: (sk.ainet.context.ExecutionContext) -> Unit,
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
        println("[DEBUG_LOG] Recording forward pass with tape-based context")

        val (tape, _) = ctx.record {
            val currentTape = this.currentTape
            if (currentTape == null) {
                error("Failed to create a recording tape for the execution context.")
            }

            // Push the current tape into the global stack for the duration of forwardPass
            val globalStack = sk.ainet.tape.Execution.tapeStack
            globalStack.pushTape(currentTape)
            println("[DEBUG_LOG] Pushed tape to global stack: $currentTape")

            try {
                forwardPass(this)
            } finally {
                globalStack.popTape()
                println("[DEBUG_LOG] Popped tape from global stack")
            }
        }

        if (tape == null) {
            error("No tape was produced during recording")
        }

        // Prefer DefaultExecutionTape.toComputeGraph() which builds a real graph from traces/ops.
        val graph = when (tape) {
            is DefaultExecutionTape -> tape.toComputeGraph()
            // Fallback to the generic extension (may be a stub in some builds)
            else -> tape.toComputeGraph()
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

        println("[INTEGRATION] Starting Arduino C code generation for library: $libraryName")
        println("[INTEGRATION] Output path: $outputPath")
        println("[INTEGRATION] Graph nodes: ${graph.nodes.size}")

        try {
            // Create and validate the C code generator
            println("[INTEGRATION] Creating CCodeGenerator and validating graph...")
            val codeGenerator = CCodeGenerator(graph)
            val validation = codeGenerator.validateGraph()
            if (validation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
                val errorMessage = "Graph validation failed: ${validation.errors.joinToString("; ")}"
                println("[INTEGRATION] ERROR: $errorMessage")
                throw IllegalArgumentException(errorMessage)
            }
            println("[INTEGRATION] Graph validation successful")

        // Generate all components needed for the Arduino library
        println("[INTEGRATION] Generating C code components...")
        val memoryLayout = codeGenerator.calculateMemoryRequirements()
        println("[INTEGRATION] Memory requirements calculated: ${memoryLayout.totalMemoryRequired} bytes total")

        val layers = codeGenerator.generateAllLayers()
        println("[INTEGRATION] Generated ${layers.size} layer code fragments")

        val weights = codeGenerator.extractWeights()
        println("[INTEGRATION] Extracted ${weights.size} weight arrays")

        // Determine input/output dimensions from the graph
        println("[INTEGRATION] Determining input/output dimensions...")
        val inputDims = determineInputDimensions(graph)
        val outputDims = determineOutputDimensions(graph)
        println("[INTEGRATION] Input dimensions: ${inputDims.joinToString("x")}")
        println("[INTEGRATION] Output dimensions: ${outputDims.joinToString("x")}")

        // Generate C code using templates
        println("[INTEGRATION] Generating C code using templates...")
        val headerCode = HeaderTemplate.generate(
            libraryName = libraryName,
            inputDims = inputDims,
            outputDims = outputDims,
            memoryRequirements = memoryLayout
        )
        println("[INTEGRATION] Header code generated (${headerCode.length} characters)")

        val sourceCode = SourceTemplate.generate(
            libraryName = libraryName,
            layers = layers,
            weights = weights
        )
        println("[INTEGRATION] Source code generated (${sourceCode.length} characters)")

        // Validate generated code for static allocation and buffer alternation
        println("[INTEGRATION] Validating generated code...")
        val staticValidation = codeGenerator.validateGeneratedCodeStaticAllocation(sourceCode)
        if (staticValidation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            val errorMessage = "Generated code static allocation validation failed: ${staticValidation.errors.joinToString("; ")}"
            println("[INTEGRATION] ERROR: $errorMessage")
            throw IllegalArgumentException(errorMessage)
        }
        println("[INTEGRATION] Static allocation validation passed")

        val bufferValidation = codeGenerator.validateGeneratedCodeBufferAlternation(sourceCode)
        if (bufferValidation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            val errorMessage = "Generated code buffer alternation validation failed: ${bufferValidation.errors.joinToString("; ")}"
            println("[INTEGRATION] ERROR: $errorMessage")
            throw IllegalArgumentException(errorMessage)
        }
        println("[INTEGRATION] Buffer alternation validation passed")

        // Package into Arduino library structure
        println("[INTEGRATION] Packaging into Arduino library structure...")
        val packager = ArduinoLibraryPackager()
        val result = packager.createLibraryStructure(
            outputPath = outputPath,
            libraryName = libraryName,
            sourceCode = sourceCode,
            headerCode = headerCode,
            memoryLayout = memoryLayout,
            inputDims = inputDims,
            outputDims = outputDims
        )

        // Real file writing (bridging while packager has placeholders)
        // writeLibraryToDisk(result, sourceCode, headerCode, libraryName)

        println("[INTEGRATION] Arduino library generation completed successfully!")
        println("[INTEGRATION] Library path: ${result.libraryPath}")
        println("[INTEGRATION] Generated files: ${result.generatedFiles.size}")
        println("[INTEGRATION] Supported operations: ${result.supportedOperations.joinToString(", ")}")

        return result

        } catch (e: Exception) {
            println("[INTEGRATION] ERROR: Arduino library generation failed: ${e.message}")
            println("[INTEGRATION] Exception type: ${e::class.simpleName}")
            throw e
        }
    }

    /**
     * Writes the generated library components to the physical disk.
     * This bridges the gap while ArduinoLibraryPackager has placeholder implementations.
     */
    private fun writeLibraryToDisk(result: ArduinoLibraryResult, sourceCode: String, headerCode: String, libraryName: String) {
        // Implementation moved to CCodeGenerator or similar if needed.
        // For now we will use a more direct approach in the facade for JVM if we can detect it.
    }

    /**
     * Determines input dimensions from the ComputeGraph.
     * 
     * @param graph The ComputeGraph to analyze
     * @return Array of input tensor dimensions
     */
    private fun determineInputDimensions(graph: ComputeGraph): IntArray {
        println("[DEBUG_LOG] determineInputDimensions for graph with ${graph.nodes.size} nodes")
        // Find input nodes (nodes with no incoming edges)
        val inputNodes = graph.getInputNodes()
        println("[DEBUG_LOG] inputNodes: ${inputNodes.map { it.id }}")

        if (inputNodes.isEmpty()) {
            throw IllegalArgumentException("Graph has no input nodes. Nodes: ${graph.nodes.map { it.id }}")
        }

        // Use the first input node's output shape as the input dimensions
        val inputNode = inputNodes.first()
        val inputShape = inputNode.outputs.firstOrNull()?.shape
            ?: throw IllegalArgumentException("Input node ${inputNode.id} has no output shape")

        return inputShape.toIntArray()
    }

    /**
     * Determines output dimensions from the ComputeGraph.
     * 
     * @param graph The ComputeGraph to analyze
     * @return Array of output tensor dimensions
     */
    private fun determineOutputDimensions(graph: ComputeGraph): IntArray {
        // Find output nodes (nodes with no outgoing edges)
        val outputNodes = graph.getOutputNodes()

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