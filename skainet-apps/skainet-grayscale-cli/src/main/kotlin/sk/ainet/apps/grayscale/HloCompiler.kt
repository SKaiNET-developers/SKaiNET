package sk.ainet.apps.grayscale

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32

/**
 * Compiler for converting grayscale models to StableHLO MLIR format.
 * Handles model execution recording, compute graph generation, and HLO compilation.
 */
public class HloCompiler {
    
    /**
     * Compiles a grayscale model to StableHLO MLIR format.
     * 
     * @param modelInstance The grayscale model instance to compile
     * @param sampleInput A sample input tensor for recording execution
     * @return CompilationResult containing the compiled module or error information
     */
    public suspend fun compileModel(
        modelInstance: GrayscaleModelInstance,
        sampleInput: LoadedImage
    ): CompilationResult {
        val startTime = System.currentTimeMillis()
        
        return try {
            when (modelInstance) {
                is GrayscaleModelInstance.FP32Model -> {
                    compileModelFP32(modelInstance, sampleInput, startTime)
                }
                is GrayscaleModelInstance.FP16Model -> {
                    compileModelFP16(modelInstance, sampleInput, startTime)
                }
            }
        } catch (e: Exception) {
            val compilationTime = System.currentTimeMillis() - startTime
            CompilationResult.Error(
                modelType = when (modelInstance) {
                    is GrayscaleModelInstance.FP32Model -> GrayscaleModelType.RGB2GRAYSCALE
                    is GrayscaleModelInstance.FP16Model -> GrayscaleModelType.RGB2GRAYSCALE_MATMUL
                },
                error = "HLO compilation failed: ${e.message}",
                compilationTimeMs = compilationTime,
                cause = e
            )
        }
    }
    
    /**
     * Compiles an FP32 model to StableHLO.
     */
    private suspend fun compileModelFP32(
        modelInstance: GrayscaleModelInstance.FP32Model,
        sampleInput: LoadedImage,
        startTime: Long
    ): CompilationResult {
        // Create a recording context to capture model execution
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        
        val (tape, _) = ctx.record {
            try {
                // Convert sample image to tensor
                val imageLoader = ImageLoader()
                val inputTensor = imageLoader.imageToTensor(sampleInput, modelInstance.executionContext)
                
                // Convert to FP32 if needed
                val fp32InputTensor = convertToFP32(inputTensor, modelInstance.executionContext)
                
                // Create model module and execute to record operations
                val module = modelInstance.model.create(modelInstance.executionContext)
                modelInstance.model.calculate(
                    module = module,
                    inputValue = fp32InputTensor,
                    executionContext = modelInstance.executionContext
                ) { _, _, _ -> 
                    // Progress callback - no action needed for compilation
                }
            } catch (e: Exception) {
                throw HloCompilationException("Failed to record model execution for FP32 model", e)
            }
        }
        
        // Convert tape to compute graph
        val computeGraph = when (tape) {
            is DefaultExecutionTape -> tape.toComputeGraph()
            else -> tape?.toComputeGraph() ?: throw HloCompilationException("Failed to create compute graph from execution tape")
        }
        
        // Compile to StableHLO
        val hloModule = try {
            toStableHlo(computeGraph, "grayscale_convert_fp32")
        } catch (e: Exception) {
            throw HloCompilationException("Failed to convert compute graph to StableHLO", e)
        }
        
        // Validate the generated HLO
        validateHloModule(hloModule)
        
        val compilationTime = System.currentTimeMillis() - startTime
        
        return CompilationResult.Success(
            modelType = GrayscaleModelType.RGB2GRAYSCALE,
            hloModule = hloModule,
            computeGraph = computeGraph,
            compilationTimeMs = compilationTime,
            inputShape = listOf(1, 3, sampleInput.height, sampleInput.width),
            outputShape = listOf(1, 1, sampleInput.height, sampleInput.width)
        )
    }
    
    /**
     * Compiles an FP16 model to StableHLO.
     */
    private suspend fun compileModelFP16(
        modelInstance: GrayscaleModelInstance.FP16Model,
        sampleInput: LoadedImage,
        startTime: Long
    ): CompilationResult {
        // Create a recording context to capture model execution
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        
        val (tape, _) = ctx.record {
            try {
                // Convert sample image to tensor
                val imageLoader = ImageLoader()
                val inputTensor = imageLoader.imageToTensor(sampleInput, modelInstance.executionContext)
                
                // Create model module and execute to record operations
                val module = modelInstance.model.create(modelInstance.executionContext)
                modelInstance.model.calculate(
                    module = module,
                    inputValue = inputTensor,
                    executionContext = modelInstance.executionContext
                ) { _, _, _ -> 
                    // Progress callback - no action needed for compilation
                }
            } catch (e: Exception) {
                throw HloCompilationException("Failed to record model execution for FP16 model", e)
            }
        }
        
        // Convert tape to compute graph
        val computeGraph = when (tape) {
            is DefaultExecutionTape -> tape.toComputeGraph()
            else -> tape?.toComputeGraph() ?: throw HloCompilationException("Failed to create compute graph from execution tape")
        }
        
        // Compile to StableHLO
        val hloModule = try {
            toStableHlo(computeGraph, "grayscale_convert_fp16")
        } catch (e: Exception) {
            throw HloCompilationException("Failed to convert compute graph to StableHLO", e)
        }
        
        // Validate the generated HLO
        validateHloModule(hloModule)
        
        val compilationTime = System.currentTimeMillis() - startTime
        
        return CompilationResult.Success(
            modelType = GrayscaleModelType.RGB2GRAYSCALE_MATMUL,
            hloModule = hloModule,
            computeGraph = computeGraph,
            compilationTimeMs = compilationTime,
            inputShape = listOf(1, 3, sampleInput.height, sampleInput.width),
            outputShape = listOf(1, 1, sampleInput.height, sampleInput.width)
        )
    }
    
    /**
     * Converts an FP16 tensor to FP32 for use with FP32 models.
     */
    private fun convertToFP32(
        tensor: Tensor<FP16, Float>,
        context: ExecutionContext
    ): Tensor<FP32, Float> {
        // Extract the float array from the tensor data
        val floatData = (tensor.data as FloatArrayTensorData<FP16>).buffer
        
        // Create a new FP32 tensor with the same shape and data
        return context.fromFloatArray<FP32, Float>(
            shape = tensor.shape,
            dtype = FP32::class,
            data = floatData
        )
    }
    
    /**
     * Validates the generated HLO module for basic correctness.
     */
    private fun validateHloModule(hloModule: StableHloModule) {
        // Basic validation - check that the module content is not empty
        if (hloModule.content.isBlank()) {
            throw HloCompilationException("Generated HLO module is empty")
        }
        
        // Check for basic MLIR structure
        if (!hloModule.content.contains("module {")) {
            throw HloCompilationException("Generated HLO module does not contain valid MLIR module structure")
        }
        
        if (!hloModule.content.contains("func.func")) {
            throw HloCompilationException("Generated HLO module does not contain function definition")
        }
        
        // Check for StableHLO operations (at least some should be present)
        val hasStableHloOps = hloModule.content.contains("stablehlo.") || 
                             hloModule.content.contains("// Unsupported op")
        
        if (!hasStableHloOps) {
            throw HloCompilationException("Generated HLO module does not contain any StableHLO operations")
        }
    }
}

/**
 * Sealed class representing the result of HLO compilation.
 */
public sealed class CompilationResult {
    public abstract val modelType: GrayscaleModelType
    public abstract val compilationTimeMs: Long
    
    public data class Success(
        override val modelType: GrayscaleModelType,
        val hloModule: StableHloModule,
        val computeGraph: ComputeGraph,
        override val compilationTimeMs: Long,
        val inputShape: List<Int>,
        val outputShape: List<Int>
    ) : CompilationResult()
    
    public data class Error(
        override val modelType: GrayscaleModelType,
        val error: String,
        override val compilationTimeMs: Long,
        val cause: Throwable? = null
    ) : CompilationResult()
}

/**
 * Exception thrown when HLO compilation fails.
 * @deprecated Use GrayscaleCliError.CompilationError instead
 */
@Deprecated("Use GrayscaleCliError.CompilationError instead", ReplaceWith("GrayscaleCliError.CompilationError"))
public class HloCompilationException(message: String, cause: Throwable? = null) : Exception(message, cause)