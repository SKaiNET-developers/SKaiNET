package sk.ainet.apps.grayscale

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tape.toComputeGraph
import sk.ainet.lang.tensor.ops.VoidTensorOps
import java.io.File

/**
 * Enum representing available backend types for execution.
 */
public enum class BackendType(public val displayName: String, public val description: String) {
    CPU("CPU", "Pure Kotlin CPU execution (default)"),
    JVM_SIMD("JVM-SIMD", "JVM with Vector API SIMD acceleration (Java 21+)"),
    HLO_EXPORT("HLO-Export", "Export compute graph to StableHLO MLIR format");

    public companion object {
        public fun fromString(value: String): BackendType = when (value.lowercase()) {
            "cpu" -> CPU
            "jvm-simd", "jvm_simd", "simd" -> JVM_SIMD
            "hlo-export", "hlo_export", "hlo", "stablehlo" -> HLO_EXPORT
            else -> throw IllegalArgumentException(
                "Unknown backend: $value. Valid options: ${values().joinToString(", ") { it.name.lowercase() }}"
            )
        }
    }
}

/**
 * Result of backend initialization.
 */
public sealed class BackendResult {
    /**
     * Backend ready for execution.
     */
    public data class Ready(
        val executionContext: ExecutionContext,
        val backendType: BackendType,
        val info: String
    ) : BackendResult()

    /**
     * HLO export mode - operations will be traced for export.
     */
    public data class HloExportMode(
        val tracingContext: DefaultGraphExecutionContext,
        val outputPath: String
    ) : BackendResult()

    /**
     * Backend initialization failed.
     */
    public data class Failed(
        val reason: String,
        val cause: Throwable? = null
    ) : BackendResult()
}

/**
 * Manages backend selection and initialization for different execution strategies.
 *
 * Supported backends:
 * - CPU: Pure Kotlin CPU execution (works everywhere)
 * - JVM-SIMD: JVM with Vector API SIMD (Java 21+ with --add-modules jdk.incubator.vector)
 * - HLO-Export: Exports compute graph to StableHLO MLIR for external compilation
 *
 * Note: MLX backend is available in native macOS builds (skainet-keyword-spotter).
 */
public class BackendManager {

    private val logger = Logger(verbose = true)

    /**
     * Creates an execution backend based on the specified type.
     *
     * @param backendType The type of backend to create
     * @param verbose Whether to output detailed information
     * @param hloOutputPath Path for HLO export (required when backendType is HLO_EXPORT)
     * @return BackendResult containing the execution context or export configuration
     */
    public fun createBackend(
        backendType: BackendType,
        verbose: Boolean = false,
        hloOutputPath: String? = null
    ): BackendResult {
        return when (backendType) {
            BackendType.CPU -> createCpuBackend(verbose)
            BackendType.JVM_SIMD -> createJvmSimdBackend(verbose)
            BackendType.HLO_EXPORT -> createHloExportBackend(hloOutputPath, verbose)
        }
    }

    /**
     * Creates a standard CPU execution context.
     */
    private fun createCpuBackend(verbose: Boolean): BackendResult {
        return try {
            // Ensure SIMD is disabled for pure CPU mode
            System.setProperty("skainet.cpu.vector.enabled", "false")

            val context = DirectCpuExecutionContext()

            if (verbose) {
                println("Backend: CPU (Pure Kotlin)")
                println("  SIMD: Disabled")
                println("  BLAS: ${System.getProperty("skainet.cpu.blas.enabled", "false")}")
            }

            BackendResult.Ready(
                executionContext = context,
                backendType = BackendType.CPU,
                info = "Pure Kotlin CPU execution"
            )
        } catch (e: Exception) {
            BackendResult.Failed("Failed to initialize CPU backend: ${e.message}", e)
        }
    }

    /**
     * Creates a JVM execution context with SIMD acceleration enabled.
     * Requires Java 21+ with Vector API support.
     */
    private fun createJvmSimdBackend(verbose: Boolean): BackendResult {
        return try {
            // Check Java version for Vector API support
            val javaVersion = System.getProperty("java.version")
            val majorVersion = javaVersion.split(".").firstOrNull()?.toIntOrNull() ?: 0

            if (majorVersion < 21) {
                return BackendResult.Failed(
                    "JVM-SIMD backend requires Java 21+. Current version: $javaVersion\n" +
                    "Please upgrade to Java 21 or use --backend=cpu"
                )
            }

            // Enable SIMD via system property
            System.setProperty("skainet.cpu.vector.enabled", "true")

            val context = DirectCpuExecutionContext()

            if (verbose) {
                println("Backend: JVM-SIMD (Vector API)")
                println("  Java Version: $javaVersion")
                println("  SIMD: Enabled")
                println("  Note: For optimal performance, run with: --add-modules jdk.incubator.vector")
            }

            BackendResult.Ready(
                executionContext = context,
                backendType = BackendType.JVM_SIMD,
                info = "JVM SIMD (Vector API) - Java $javaVersion"
            )
        } catch (e: Exception) {
            BackendResult.Failed("Failed to initialize JVM-SIMD backend: ${e.message}", e)
        }
    }

    /**
     * Creates an HLO export configuration for tracing operations.
     */
    private fun createHloExportBackend(outputPath: String?, verbose: Boolean): BackendResult {
        if (outputPath == null) {
            return BackendResult.Failed(
                "HLO export requires --hlo-output path to be specified"
            )
        }

        return try {
            // Validate output path
            val outputFile = File(outputPath)
            val parentDir = outputFile.parentFile
            if (parentDir != null && !parentDir.exists()) {
                if (!parentDir.mkdirs()) {
                    return BackendResult.Failed(
                        "Cannot create output directory: ${parentDir.absolutePath}"
                    )
                }
            }

            // Create tracing context with VoidTensorOps for shape inference only
            val tracingContext = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())

            if (verbose) {
                println("Backend: HLO-Export (StableHLO MLIR)")
                println("  Output: $outputPath")
                println("  Mode: Trace operations and export to MLIR")
            }

            BackendResult.HloExportMode(
                tracingContext = tracingContext,
                outputPath = outputPath
            )
        } catch (e: Exception) {
            BackendResult.Failed("Failed to initialize HLO export: ${e.message}", e)
        }
    }

    /**
     * Exports a recorded execution tape to StableHLO MLIR format.
     *
     * @param tape The execution tape containing recorded operations
     * @param functionName Name for the exported function
     * @param outputPath Path to write the MLIR file
     * @return The generated StableHloModule or null if export failed
     */
    public fun exportToHlo(
        tape: DefaultExecutionTape,
        functionName: String,
        outputPath: String
    ): StableHloModule? {
        return try {
            // Convert tape to compute graph
            val computeGraph = tape.toComputeGraph()

            // Export to StableHLO
            val hloModule = toStableHlo(computeGraph, functionName)

            // Write to file
            File(outputPath).writeText(hloModule.content)

            println("StableHLO exported successfully!")
            println("  Function: ${hloModule.functionName}")
            println("  Inputs: ${hloModule.inputSpecs.size}")
            println("  Outputs: ${hloModule.outputSpecs.size}")
            println("  File: $outputPath")

            hloModule
        } catch (e: Exception) {
            println("HLO export failed: ${e.message}")
            null
        }
    }

    /**
     * Returns information about available backends.
     */
    public fun getBackendInfo(): String = buildString {
        appendLine("Available backends:")
        BackendType.values().forEach { backend ->
            appendLine("  ${backend.name.lowercase().padEnd(12)} - ${backend.description}")
        }
        appendLine()
        appendLine("Notes:")
        appendLine("  - JVM-SIMD requires Java 21+ and --add-modules jdk.incubator.vector")
        appendLine("  - HLO-Export generates StableHLO MLIR for external compilers (IREE, XLA)")
        appendLine("  - MLX backend available in native macOS builds (skainet-keyword-spotter)")
    }
}
