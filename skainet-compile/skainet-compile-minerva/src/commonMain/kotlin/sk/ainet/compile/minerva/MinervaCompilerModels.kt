package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportContext

/**
 * Platform defaults for compiler execution and project packaging.
 */
public expect object MinervaPlatformExportDefaults {
    public fun compilerAdapter(): MinervaCompilerAdapter
    public fun projectPackager(): MinervaProjectPackager
}

/**
 * Compiler invocation input after lowering and NPZ emission.
 */
public data class MinervaCompilerRequest(
    public val options: MinervaExportOptions,
    public val intermediate: MinervaIntermediate,
    public val npzModel: MinervaNpzModel
) {
    init {
        require(options.projectName == intermediate.projectName) {
            "compiler request options and intermediate project names must match"
        }
    }
}

/**
 * Paths and diagnostics returned by a successful Minerva compiler invocation.
 */
public data class MinervaCompilerOutput(
    public val outputDir: String,
    public val weightsCPath: String,
    public val weightsHPath: String,
    public val debugWeightsPath: String? = null,
    public val commandSummary: String,
    public val stdout: String = "",
    public val stderr: String = "",
    public val exitCode: Int = 0,
    public val generatedFiles: List<String> = listOf(weightsCPath, weightsHPath) +
        listOfNotNull(debugWeightsPath),
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(outputDir.isNotBlank()) { "compiler outputDir cannot be blank" }
        require(weightsCPath.isNotBlank()) { "weightsCPath cannot be blank" }
        require(weightsHPath.isNotBlank()) { "weightsHPath cannot be blank" }
        require(debugWeightsPath == null || debugWeightsPath.isNotBlank()) {
            "debugWeightsPath cannot be blank when provided"
        }
        require(commandSummary.isNotBlank()) { "commandSummary cannot be blank" }
        require(exitCode == 0) { "successful compiler output must have exitCode 0" }
        require(generatedFiles.all { it.isNotBlank() }) { "generatedFiles cannot contain blank paths" }
    }
}

/**
 * Typed compiler error that can be mapped into [MinervaExportFailure].
 */
public class MinervaCompilerException(
    message: String,
    public val code: String,
    public val prerequisite: Boolean = false,
    public val stdout: String = "",
    public val stderr: String = "",
    public val exitCode: Int? = null,
    public val commandSummary: String? = null,
    public val remediation: String,
    public val details: Map<String, String> = emptyMap()
) : IllegalStateException(message) {
    init {
        require(code.isNotBlank()) { "compiler exception code cannot be blank" }
        require(remediation.isNotBlank()) { "compiler exception remediation cannot be blank" }
    }
}

/**
 * Invokes libminerva compiler tooling for a lowered model.
 */
public interface MinervaCompilerAdapter {
    public val backendName: String

    public fun compile(
        request: MinervaCompilerRequest,
        context: GraphExportContext
    ): MinervaCompilerOutput
}

/**
 * Packaging input produced after successful compiler invocation.
 */
public data class MinervaProjectPackageRequest(
    public val options: MinervaExportOptions,
    public val intermediate: MinervaIntermediate,
    public val npzModel: MinervaNpzModel,
    public val compilerOutput: MinervaCompilerOutput
)

/**
 * Typed packaging error that can be mapped into [MinervaExportFailure].
 */
public class MinervaPackagingException(
    message: String,
    public val code: String,
    public val remediation: String,
    public val details: Map<String, String> = emptyMap()
) : IllegalStateException(message) {
    init {
        require(code.isNotBlank()) { "packaging exception code cannot be blank" }
        require(remediation.isNotBlank()) { "packaging exception remediation cannot be blank" }
    }
}

/**
 * Packages compiler outputs into a Minerva project directory.
 */
public interface MinervaProjectPackager {
    public val backendName: String

    public fun packageProject(
        request: MinervaProjectPackageRequest,
        context: GraphExportContext
    ): MinervaExportBundle
}
