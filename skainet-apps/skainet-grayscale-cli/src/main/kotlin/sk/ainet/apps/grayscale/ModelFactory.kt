package sk.ainet.apps.grayscale

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32

/**
 * Factory for creating grayscale conversion models based on configuration.
 * Supports both Rgb2GrayScale and Rgb2GrayScaleMatMul models with appropriate execution contexts.
 */
public class ModelFactory {
    
    private val executionContextManager = ExecutionContextManager()
    
    /**
     * Creates a grayscale conversion model based on the specified type.
     * 
     * @param modelType The type of grayscale model to create
     * @param useGpu Whether to use GPU acceleration (falls back to CPU if unavailable)
     * @param verbose Whether to output detailed context selection information
     * @return A pair of the model and its execution context
     */
    public fun createGrayscaleModel(
        modelType: GrayscaleModelType,
        useGpu: Boolean = false,
        verbose: Boolean = false
    ): GrayscaleModelInstance {
        // Create execution context using the context manager
        val contextResult = executionContextManager.createExecutionContext(useGpu, verbose)
        
        // Provide dependency guidance if GPU was requested but not available
        if (useGpu && contextResult.fallbackReason != null) {
            val guidance = executionContextManager.provideDependencyGuidance(contextResult.gpuCapabilities)
            if (guidance.isNotEmpty() && verbose) {
                println("GPU Dependency Guidance:")
                guidance.forEach { println("  $it") }
                println()
            }
        }
        
        return when (modelType) {
            GrayscaleModelType.RGB2GRAYSCALE -> {
                val model = Rgb2GrayScale()
                GrayscaleModelInstance.FP32Model(model, contextResult.executionContext)
            }
            GrayscaleModelType.RGB2GRAYSCALE_MATMUL -> {
                val model = Rgb2GrayScaleMatMul(contextResult.executionContext)
                GrayscaleModelInstance.FP16Model(model, contextResult.executionContext)
            }
        }
    }
    
    /**
     * Gets GPU capabilities information for the current system.
     * 
     * @return GpuCapabilities containing system GPU information
     */
    public fun getGpuCapabilities(): GpuCapabilities {
        return executionContextManager.createExecutionContext(preferGpu = false).gpuCapabilities
    }
}

/**
 * Sealed class representing different model instances with their type information.
 * This handles the fact that different models use different precision types.
 */
public sealed class GrayscaleModelInstance {
    public abstract val executionContext: ExecutionContext
    
    /**
     * FP32-based model instance (Rgb2GrayScale)
     */
    public data class FP32Model(
        val model: Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>>,
        override val executionContext: ExecutionContext
    ) : GrayscaleModelInstance()
    
    /**
     * FP16-based model instance (Rgb2GrayScaleMatMul)
     */
    public data class FP16Model(
        val model: Model<FP16, Float, Tensor<FP16, Float>, Tensor<FP16, Float>>,
        override val executionContext: ExecutionContext
    ) : GrayscaleModelInstance()
}