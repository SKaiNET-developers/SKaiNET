package sk.ainet.apps.grayscale

import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Validates system dependencies and platform compatibility for the Grayscale CLI application.
 * Provides comprehensive checks for CUDA drivers, system libraries, and platform-specific requirements.
 */
class DependencyValidator {
    
    /**
     * Performs a comprehensive validation of all system dependencies.
     * 
     * @param requireGpu Whether GPU capabilities are required (vs optional)
     * @param verbose Whether to output detailed validation information
     * @return ValidationResult containing the overall status and any issues found
     */
    fun validateAllDependencies(requireGpu: Boolean = false, verbose: Boolean = false): ValidationResult {
        val issues = mutableListOf<GrayscaleCliError.SystemError>()
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Performing system dependency validation...")
            println("Platform: ${System.getProperty("os.name")} ${System.getProperty("os.arch")}")
            println("Java Version: ${System.getProperty("java.version")}")
            println()
        }
        
        // Validate platform compatibility
        val platformResult = validatePlatform(verbose)
        if (platformResult.error != null) {
            issues.add(platformResult.error)
        }
        warnings.addAll(platformResult.warnings)
        
        // Validate Java environment
        val javaResult = validateJavaEnvironment(verbose)
        if (javaResult.error != null) {
            issues.add(javaResult.error)
        }
        warnings.addAll(javaResult.warnings)
        
        // Validate CUDA (if GPU is requested or required)
        if (requireGpu) {
            val cudaResult = validateCudaDrivers(verbose)
            if (cudaResult.error != null) {
                issues.add(cudaResult.error)
            }
            warnings.addAll(cudaResult.warnings)
        } else {
            // Optional GPU validation - only warn if issues found
            val cudaResult = validateCudaDrivers(verbose, optional = true)
            warnings.addAll(cudaResult.warnings)
        }
        
        // Validate system resources
        val resourceResult = validateSystemResources(verbose)
        if (resourceResult.error != null) {
            issues.add(resourceResult.error)
        }
        warnings.addAll(resourceResult.warnings)
        
        // Validate SKaiNET dependencies
        val skainetResult = validateSkainetDependencies(verbose)
        if (skainetResult.error != null) {
            issues.add(skainetResult.error)
        }
        warnings.addAll(skainetResult.warnings)
        
        val success = issues.isEmpty()
        
        if (verbose) {
            if (success) {
                println("✓ All dependency validations passed")
            } else {
                println("✗ Found ${issues.size} critical issue(s)")
            }
            if (warnings.isNotEmpty()) {
                println("⚠ Found ${warnings.size} warning(s)")
            }
            println()
        }
        
        return ValidationResult(
            success = success,
            errors = issues,
            warnings = warnings,
            platformInfo = getPlatformInfo()
        )
    }
    
    /**
     * Validates platform compatibility.
     */
    private fun validatePlatform(verbose: Boolean): ComponentValidationResult {
        val osName = System.getProperty("os.name").lowercase()
        val osArch = System.getProperty("os.arch").lowercase()
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Validating platform compatibility...")
        }
        
        // Check for supported operating systems
        val supportedOS = when {
            osName.contains("windows") -> true
            osName.contains("linux") -> true
            osName.contains("mac") -> true
            else -> false
        }
        
        if (!supportedOS) {
            return ComponentValidationResult(
                error = GrayscaleCliError.SystemError.UnsupportedPlatform(
                    platform = "$osName ($osArch)",
                    feature = "SKaiNET Grayscale CLI"
                )
            )
        }
        
        // Check architecture
        val supportedArch = when {
            osArch.contains("x86_64") || osArch.contains("amd64") -> true
            osArch.contains("aarch64") || osArch.contains("arm64") -> true
            else -> false
        }
        
        if (!supportedArch) {
            warnings.add("Architecture $osArch may have limited support")
        }
        
        if (verbose) {
            println("  ✓ Platform: $osName ($osArch) - supported")
        }
        
        return ComponentValidationResult(warnings = warnings)
    }
    
    /**
     * Validates Java environment requirements.
     */
    private fun validateJavaEnvironment(verbose: Boolean): ComponentValidationResult {
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Validating Java environment...")
        }
        
        // Check Java version
        val javaVersion = System.getProperty("java.version")
        val javaMajorVersion = try {
            val versionParts = javaVersion.split(".")
            if (versionParts[0] == "1") {
                versionParts[1].toInt() // Java 8 format: 1.8.x
            } else {
                versionParts[0].toInt() // Java 9+ format: 11.x.x
            }
        } catch (e: Exception) {
            return ComponentValidationResult(
                error = GrayscaleCliError.SystemError.ConfigurationError(
                    component = "Java Runtime",
                    issue = "Cannot determine Java version: $javaVersion",
                    fixSuggestions = listOf(
                        "Ensure you're running a supported Java version (11 or higher)",
                        "Check your JAVA_HOME environment variable",
                        "Reinstall Java if necessary"
                    )
                )
            )
        }
        
        if (javaMajorVersion < 11) {
            return ComponentValidationResult(
                error = GrayscaleCliError.SystemError.MissingDependency(
                    dependency = "Java 11+",
                    purpose = "SKaiNET runtime requirements",
                    installationGuide = listOf(
                        "Install Java 11 or higher:",
                        "  - Download from: https://adoptium.net/",
                        "  - Or use your system package manager",
                        "  - Set JAVA_HOME environment variable",
                        "  - Verify with: java -version"
                    )
                )
            )
        }
        
        // Check available memory
        val maxMemory = Runtime.getRuntime().maxMemory()
        val maxMemoryMB = maxMemory / (1024 * 1024)
        
        if (maxMemoryMB < 512) {
            warnings.add("Low JVM heap memory: ${maxMemoryMB}MB (recommend 1GB+ for large images)")
        }
        
        if (verbose) {
            println("  ✓ Java Version: $javaVersion (major: $javaMajorVersion)")
            println("  ✓ Max Heap Memory: ${maxMemoryMB}MB")
        }
        
        return ComponentValidationResult(warnings = warnings)
    }
    
    /**
     * Validates CUDA driver availability and compatibility.
     */
    private fun validateCudaDrivers(verbose: Boolean, optional: Boolean = false): ComponentValidationResult {
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Validating CUDA drivers...")
        }
        
        // Check for nvidia-smi command
        val nvidiaSmiResult = executeCommand("nvidia-smi", "--version")
        
        if (!nvidiaSmiResult.success) {
            val message = "CUDA drivers not detected (nvidia-smi not found)"
            
            if (optional) {
                warnings.add("$message - GPU acceleration will not be available")
                if (verbose) {
                    println("  ⚠ $message")
                }
                return ComponentValidationResult(warnings = warnings)
            } else {
                return ComponentValidationResult(
                    error = GrayscaleCliError.SystemError.MissingDependency(
                        dependency = "NVIDIA CUDA Drivers",
                        purpose = "GPU acceleration",
                        installationGuide = listOf(
                            "Install NVIDIA CUDA Toolkit and drivers:",
                            "  1. Download from: https://developer.nvidia.com/cuda-toolkit",
                            "  2. Install appropriate version for your GPU",
                            "  3. Add CUDA to your PATH environment variable",
                            "  4. Verify installation with: nvidia-smi",
                            "  5. Restart your system after installation"
                        )
                    )
                )
            }
        }
        
        // Parse CUDA version if available
        val cudaVersion = parseCudaVersion(nvidiaSmiResult.output)
        
        if (verbose) {
            println("  ✓ NVIDIA drivers detected")
            if (cudaVersion != null) {
                println("  ✓ CUDA Version: $cudaVersion")
            }
        }
        
        // Check for CUDA runtime libraries (placeholder for future implementation)
        val cudaRuntimeAvailable = checkCudaRuntime()
        if (!cudaRuntimeAvailable) {
            warnings.add("CUDA runtime libraries may not be properly installed")
        }
        
        return ComponentValidationResult(warnings = warnings)
    }
    
    /**
     * Validates system resources (memory, disk space).
     */
    private fun validateSystemResources(verbose: Boolean): ComponentValidationResult {
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Validating system resources...")
        }
        
        // Check available memory
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        val availableMemory = maxMemory - (totalMemory - freeMemory)
        val availableMemoryMB = availableMemory / (1024 * 1024)
        
        if (availableMemoryMB < 256) {
            return ComponentValidationResult(
                error = GrayscaleCliError.SystemError.ConfigurationError(
                    component = "System Memory",
                    issue = "Insufficient available memory: ${availableMemoryMB}MB",
                    fixSuggestions = listOf(
                        "Close other applications to free memory",
                        "Increase JVM heap size with -Xmx flag",
                        "Process smaller images or reduce batch size",
                        "Consider adding more system RAM"
                    )
                )
            )
        }
        
        if (availableMemoryMB < 512) {
            warnings.add("Low available memory: ${availableMemoryMB}MB (recommend 1GB+ for optimal performance)")
        }
        
        // Check disk space in current directory
        val currentDir = File(".")
        val freeSpace = currentDir.freeSpace
        val freeSpaceMB = freeSpace / (1024 * 1024)
        
        if (freeSpaceMB < 100) {
            warnings.add("Low disk space: ${freeSpaceMB}MB (may affect output file creation)")
        }
        
        if (verbose) {
            println("  ✓ Available Memory: ${availableMemoryMB}MB")
            println("  ✓ Free Disk Space: ${freeSpaceMB}MB")
        }
        
        return ComponentValidationResult(warnings = warnings)
    }
    
    /**
     * Validates SKaiNET-specific dependencies.
     */
    private fun validateSkainetDependencies(verbose: Boolean): ComponentValidationResult {
        val warnings = mutableListOf<String>()
        
        if (verbose) {
            println("Validating SKaiNET dependencies...")
        }
        
        // Check for required SKaiNET classes (basic validation)
        val requiredClasses = listOf(
            "sk.ainet.context.ExecutionContext",
            "sk.ainet.lang.model.Model",
            "sk.ainet.io.image.PlatformBitmapImage"
        )
        
        for (className in requiredClasses) {
            try {
                Class.forName(className)
            } catch (e: ClassNotFoundException) {
                return ComponentValidationResult(
                    error = GrayscaleCliError.SystemError.MissingDependency(
                        dependency = "SKaiNET Core Libraries",
                        purpose = "neural network operations",
                        installationGuide = listOf(
                            "Ensure SKaiNET is properly installed:",
                            "  1. Check that all SKaiNET JAR files are in classpath",
                            "  2. Verify the installation is complete",
                            "  3. Rebuild the application if necessary",
                            "Missing class: $className"
                        )
                    )
                )
            }
        }
        
        if (verbose) {
            println("  ✓ SKaiNET core classes available")
        }
        
        return ComponentValidationResult(warnings = warnings)
    }
    
    /**
     * Executes a system command and returns the result.
     */
    private fun executeCommand(vararg command: String): CommandResult {
        return try {
            val process = ProcessBuilder(*command)
                .redirectErrorStream(true)
                .start()
            
            val completed = process.waitFor(5, TimeUnit.SECONDS)
            if (!completed) {
                process.destroyForcibly()
                return CommandResult(false, "Command timed out")
            }
            
            val output = process.inputStream.bufferedReader().readText()
            CommandResult(process.exitValue() == 0, output)
        } catch (e: Exception) {
            CommandResult(false, "Command execution failed: ${e.message}")
        }
    }
    
    /**
     * Parses CUDA version from nvidia-smi output.
     */
    private fun parseCudaVersion(output: String): String? {
        return try {
            val regex = Regex("CUDA Version: ([0-9.]+)")
            regex.find(output)?.groupValues?.get(1)
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Checks for CUDA runtime libraries (placeholder implementation).
     */
    private fun checkCudaRuntime(): Boolean {
        // Placeholder for future implementation
        // In a real implementation, this would check for:
        // - libcudart.so (Linux)
        // - cudart64_*.dll (Windows)
        // - CUDA runtime in standard locations
        return true
    }
    
    /**
     * Gets comprehensive platform information.
     */
    private fun getPlatformInfo(): PlatformInfo {
        val runtime = Runtime.getRuntime()
        
        return PlatformInfo(
            osName = System.getProperty("os.name"),
            osVersion = System.getProperty("os.version"),
            osArch = System.getProperty("os.arch"),
            javaVersion = System.getProperty("java.version"),
            javaVendor = System.getProperty("java.vendor"),
            availableProcessors = runtime.availableProcessors(),
            maxMemoryMB = runtime.maxMemory() / (1024 * 1024),
            totalMemoryMB = runtime.totalMemory() / (1024 * 1024),
            freeMemoryMB = runtime.freeMemory() / (1024 * 1024)
        )
    }
    
    /**
     * Provides installation guidance for missing dependencies.
     */
    fun getInstallationGuidance(dependency: String): List<String> {
        return when (dependency.lowercase()) {
            "cuda", "nvidia-driver", "nvidia drivers" -> listOf(
                "Install NVIDIA CUDA Toolkit:",
                "  1. Visit: https://developer.nvidia.com/cuda-toolkit",
                "  2. Download the appropriate version for your OS",
                "  3. Run the installer with administrator privileges",
                "  4. Add CUDA bin directory to your PATH",
                "  5. Verify installation: nvidia-smi",
                "  6. Reboot your system"
            )
            "java", "java 11", "jdk" -> listOf(
                "Install Java 11 or higher:",
                "  1. Download from: https://adoptium.net/",
                "  2. Or use package manager:",
                "     - Ubuntu/Debian: sudo apt install openjdk-11-jdk",
                "     - CentOS/RHEL: sudo yum install java-11-openjdk-devel",
                "     - macOS: brew install openjdk@11",
                "     - Windows: Use the installer from adoptium.net",
                "  3. Set JAVA_HOME environment variable",
                "  4. Verify: java -version"
            )
            else -> listOf(
                "Check the official documentation for $dependency",
                "Ensure all system dependencies are properly installed",
                "Verify your system meets the minimum requirements"
            )
        }
    }
}

/**
 * Result of dependency validation.
 */
data class ValidationResult(
    val success: Boolean,
    val errors: List<GrayscaleCliError.SystemError>,
    val warnings: List<String>,
    val platformInfo: PlatformInfo
)

/**
 * Result of validating a specific component.
 */
private data class ComponentValidationResult(
    val error: GrayscaleCliError.SystemError? = null,
    val warnings: List<String> = emptyList()
)

/**
 * Result of executing a system command.
 */
private data class CommandResult(
    val success: Boolean,
    val output: String
)

/**
 * Comprehensive platform information.
 */
data class PlatformInfo(
    val osName: String,
    val osVersion: String,
    val osArch: String,
    val javaVersion: String,
    val javaVendor: String,
    val availableProcessors: Int,
    val maxMemoryMB: Long,
    val totalMemoryMB: Long,
    val freeMemoryMB: Long
)