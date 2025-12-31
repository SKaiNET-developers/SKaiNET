package sk.ainet.apps.grayscale

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext

/**
 * Manages execution context selection and GPU capability detection.
 * Provides automatic fallback from GPU to CPU execution when GPU is unavailable.
 * Integrates with dependency validation for comprehensive system checks.
 */
public class ExecutionContextManager {
    
    private val dependencyValidator = DependencyValidator()
    
    /**
     * Creates an execution context based on preferences and system capabilities.
     * 
     * @param preferGpu Whether to prefer GPU execution over CPU
     * @param verbose Whether to output detailed information about context selection
     * @return ExecutionContextResult containing the selected context and metadata
     */
    public fun createExecutionContext(
        preferGpu: Boolean = false,
        verbose: Boolean = false
    ): ExecutionContextResult {
        val startTime = System.currentTimeMillis()
        
        // Validate dependencies first
        val validationResult = dependencyValidator.validateAllDependencies(
            requireGpu = preferGpu,
            verbose = verbose
        )
        
        // If there are critical errors and GPU is required, throw an error
        if (!validationResult.success && preferGpu) {
            val firstError = validationResult.errors.first()
            throw firstError
        }
        
        // Detect GPU capabilities
        val gpuCapabilities = detectGpuCapabilities(validationResult)
        
        if (verbose) {
            logGpuCapabilities(gpuCapabilities)
            if (validationResult.warnings.isNotEmpty()) {
                println("System Warnings:")
                validationResult.warnings.forEach { warning ->
                    println("  ⚠ $warning")
                }
                println()
            }
        }
        
        // Determine execution context based on preferences and capabilities
        val contextSelection = selectExecutionContext(preferGpu, gpuCapabilities, verbose)
        
        val selectionTime = System.currentTimeMillis() - startTime
        
        return ExecutionContextResult(
            executionContext = contextSelection.context,
            contextType = contextSelection.type,
            gpuCapabilities = gpuCapabilities,
            fallbackReason = contextSelection.fallbackReason,
            selectionTimeMs = selectionTime,
            validationResult = validationResult
        )
    }
    
    /**
     * Detects GPU capabilities on the current system using dependency validation.
     */
    private fun detectGpuCapabilities(validationResult: ValidationResult? = null): GpuCapabilities {
        val validation = validationResult ?: dependencyValidator.validateAllDependencies(requireGpu = false)
        
        return try {
            // Use validation results to determine GPU capabilities
            val cudaAvailable = !validation.errors.any { error ->
                error is GrayscaleCliError.SystemError.MissingDependency && 
                error.dependency.contains("CUDA", ignoreCase = true)
            }
            
            val memoryMB = if (cudaAvailable) estimateGpuMemory() else 0L
            val computeCapability = if (cudaAvailable) detectComputeCapability() else "N/A"
            val ireeSupported = cudaAvailable && checkIreeSupport()
            
            GpuCapabilities(
                cudaAvailable = cudaAvailable,
                memoryMB = memoryMB,
                computeCapability = computeCapability,
                ireeSupported = ireeSupported,
                validationErrors = validation.errors,
                validationWarnings = validation.warnings
            )
        } catch (e: Exception) {
            // If detection fails, assume no GPU capabilities
            GpuCapabilities(
                cudaAvailable = false,
                memoryMB = 0L,
                computeCapability = "N/A",
                ireeSupported = false,
                validationErrors = validation.errors,
                validationWarnings = validation.warnings + "GPU detection failed: ${e.message}"
            )
        }
    }
    
    /**
     * Placeholder for CUDA availability detection.
     * Now uses system command execution for more reliable detection.
     */
    private fun detectCudaAvailability(): Boolean {
        return try {
            val process = ProcessBuilder("nvidia-smi", "--version")
                .redirectErrorStream(true)
                .start()
            
            val completed = process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)
            if (!completed) {
                process.destroyForcibly()
                return false
            }
            
            process.exitValue() == 0
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Placeholder for GPU memory estimation.
     */
    private fun estimateGpuMemory(): Long {
        // Placeholder implementation
        // Real implementation would query GPU memory via CUDA or nvidia-ml-py equivalent
        return 0L
    }
    
    /**
     * Placeholder for compute capability detection.
     */
    private fun detectComputeCapability(): String {
        // Placeholder implementation
        // Real implementation would query GPU compute capability
        return "N/A"
    }
    
    /**
     * Placeholder for IREE support verification.
     */
    private fun checkIreeSupport(): Boolean {
        // Placeholder implementation
        // Real implementation would check:
        // - IREE runtime availability
        // - IREE CUDA backend support
        // - Basic IREE compilation test
        return false
    }
    
    /**
     * Selects the appropriate execution context based on preferences and capabilities.
     */
    private fun selectExecutionContext(
        preferGpu: Boolean,
        gpuCapabilities: GpuCapabilities,
        verbose: Boolean
    ): ContextSelection {
        return when {
            !preferGpu -> {
                // User explicitly wants CPU execution
                if (verbose) {
                    println("Using CPU execution context (user preference)")
                }
                ContextSelection(
                    context = DirectCpuExecutionContext(),
                    type = ExecutionContextType.CPU,
                    fallbackReason = null
                )
            }
            
            !gpuCapabilities.cudaAvailable -> {
                // GPU preferred but CUDA not available
                val reason = "CUDA runtime not available"
                if (verbose) {
                    println("Warning: GPU execution requested but $reason. Falling back to CPU.")
                }
                ContextSelection(
                    context = DirectCpuExecutionContext(),
                    type = ExecutionContextType.CPU,
                    fallbackReason = reason
                )
            }
            
            !gpuCapabilities.ireeSupported -> {
                // CUDA available but IREE not supported
                val reason = "IREE GPU backend not supported"
                if (verbose) {
                    println("Warning: GPU execution requested but $reason. Falling back to CPU.")
                }
                ContextSelection(
                    context = DirectCpuExecutionContext(),
                    type = ExecutionContextType.CPU,
                    fallbackReason = reason
                )
            }
            
            gpuCapabilities.memoryMB < 512 -> {
                // GPU available but insufficient memory
                val reason = "Insufficient GPU memory (${gpuCapabilities.memoryMB}MB < 512MB required)"
                if (verbose) {
                    println("Warning: GPU execution requested but $reason. Falling back to CPU.")
                }
                ContextSelection(
                    context = DirectCpuExecutionContext(),
                    type = ExecutionContextType.CPU,
                    fallbackReason = reason
                )
            }
            
            else -> {
                // GPU execution should be possible
                // For now, still fall back to CPU since IREE integration is not yet implemented
                val reason = "IREE GPU execution not yet implemented"
                if (verbose) {
                    println("Warning: GPU capabilities detected but $reason. Falling back to CPU.")
                    println("GPU will be supported in future versions with IREE integration.")
                }
                ContextSelection(
                    context = DirectCpuExecutionContext(),
                    type = ExecutionContextType.CPU,
                    fallbackReason = reason
                )
            }
        }
    }
    
    /**
     * Logs GPU capabilities information.
     */
    private fun logGpuCapabilities(capabilities: GpuCapabilities) {
        println("GPU Capabilities Detection:")
        println("  CUDA Available: ${capabilities.cudaAvailable}")
        if (capabilities.cudaAvailable) {
            println("  GPU Memory: ${capabilities.memoryMB}MB")
            println("  Compute Capability: ${capabilities.computeCapability}")
        }
        println("  IREE Supported: ${capabilities.ireeSupported}")
        println()
    }
    
    /**
     * Provides comprehensive guidance for installing missing GPU dependencies.
     */
    public fun provideDependencyGuidance(capabilities: GpuCapabilities): List<String> {
        val guidance = mutableListOf<String>()
        
        // Add guidance based on validation errors
        capabilities.validationErrors.forEach { error ->
            when (error) {
                is GrayscaleCliError.SystemError.MissingDependency -> {
                    guidance.add("Missing: ${error.dependency}")
                    guidance.addAll(error.installationGuide)
                }
                is GrayscaleCliError.SystemError.DriverIssue -> {
                    guidance.add("Driver Issue: ${error.driver} - ${error.issue}")
                    guidance.addAll(error.installationGuide)
                }
                is GrayscaleCliError.SystemError.UnsupportedPlatform -> {
                    guidance.add("Platform Issue: ${error.platform} does not support ${error.feature}")
                    guidance.add("Consider using CPU-only execution")
                }
                is GrayscaleCliError.SystemError.ConfigurationError -> {
                    guidance.add("Configuration Issue: ${error.component} - ${error.issue}")
                    guidance.addAll(error.fixSuggestions)
                }
            }
            guidance.add("") // Add blank line between different errors
        }
        
        // Add guidance based on warnings
        if (capabilities.validationWarnings.isNotEmpty()) {
            guidance.add("Warnings:")
            capabilities.validationWarnings.forEach { warning ->
                guidance.add("  • $warning")
            }
        }
        
        // Legacy guidance for backward compatibility
        if (!capabilities.cudaAvailable && capabilities.validationErrors.isEmpty()) {
            guidance.addAll(dependencyValidator.getInstallationGuidance("CUDA"))
        }
        
        if (!capabilities.ireeSupported && capabilities.cudaAvailable) {
            guidance.add("IREE GPU support will be available in future versions")
            guidance.add("Current version supports CPU execution only")
        }
        
        if (capabilities.memoryMB in 1..511) {
            guidance.add("GPU memory is limited (${capabilities.memoryMB}MB)")
            guidance.add("Consider using CPU execution for large images")
            guidance.add("Or reduce batch size if processing multiple images")
        }
        
        return guidance.filter { it.isNotBlank() }
    }
}

/**
 * Data class representing GPU capabilities on the current system.
 */
public data class GpuCapabilities(
    val cudaAvailable: Boolean,
    val memoryMB: Long,
    val computeCapability: String,
    val ireeSupported: Boolean,
    val validationErrors: List<GrayscaleCliError.SystemError> = emptyList(),
    val validationWarnings: List<String> = emptyList()
)

/**
 * Enum representing different execution context types.
 */
public enum class ExecutionContextType {
    CPU,
    GPU_CUDA,
    GPU_VULKAN
}

/**
 * Data class representing the result of execution context creation.
 */
public data class ExecutionContextResult(
    val executionContext: ExecutionContext,
    val contextType: ExecutionContextType,
    val gpuCapabilities: GpuCapabilities,
    val fallbackReason: String?,
    val selectionTimeMs: Long,
    val validationResult: ValidationResult? = null
)

/**
 * Internal data class for context selection logic.
 */
private data class ContextSelection(
    val context: ExecutionContext,
    val type: ExecutionContextType,
    val fallbackReason: String?
)