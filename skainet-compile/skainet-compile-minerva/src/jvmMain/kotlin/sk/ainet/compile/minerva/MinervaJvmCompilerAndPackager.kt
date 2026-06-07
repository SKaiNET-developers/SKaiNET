package sk.ainet.compile.minerva

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import sk.ainet.compile.export.GraphExportArtifact
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage

public actual object MinervaPlatformExportDefaults {
    public actual fun compilerAdapter(): MinervaCompilerAdapter = PythonMinervaCompilerAdapter()

    public actual fun projectPackager(): MinervaProjectPackager = JvmMinervaProjectPackager()
}

/**
 * JVM adapter that invokes the Python libminerva compiler entry point.
 */
public class PythonMinervaCompilerAdapter @kotlin.jvm.JvmOverloads constructor(
    override val backendName: String = MinervaExportBackend.backendName
) : MinervaCompilerAdapter {

    override fun compile(
        request: MinervaCompilerRequest,
        context: GraphExportContext
    ): MinervaCompilerOutput {
        val options = request.options
        val compilerScript = options.compilerScript
            ?: prerequisiteFailure(
                code = "minerva.compiler.script_missing",
                message = "Minerva compiler script path is required before compiler invocation.",
                remediation = "Set MinervaExportOptions.compilerScript to the libminerva compiler Python entry point."
            )
        val scriptPath = requireRegularFile(
            field = "compilerScript",
            value = compilerScript,
            code = "minerva.compiler.script_not_found",
            remediation = "Point compilerScript at an existing libminerva compiler Python file."
        )
        options.runtimeRoot?.let {
            requireDirectory(
                field = "runtimeRoot",
                value = it,
                code = "minerva.compiler.runtime_root_not_found",
                remediation = "Point runtimeRoot at an existing libminerva checkout or install directory."
            )
        }
        options.keyFile?.let {
            requireRegularFile(
                field = "keyFile",
                value = it,
                code = "minerva.compiler.key_file_not_found",
                remediation = "Point keyFile at an existing key file, or omit it for non-secure local compiler tests."
            )
        }
        options.calibrationNpz?.let {
            requireRegularFile(
                field = "calibrationNpz",
                value = it,
                code = "minerva.compiler.calibration_not_found",
                remediation = "Point calibrationNpz at an existing calibration archive, or omit it when not required."
            )
        }

        val projectDir = Paths.get(options.outputDir).resolve(options.projectName).normalize()
        val generatedDir = projectDir.resolve("generated")
        val modelPath = generatedDir.resolve(request.npzModel.logicalPath.substringAfterLast('/'))
        try {
            Files.createDirectories(generatedDir)
            Files.write(modelPath, request.npzModel.bytes)
        } catch (exception: IOException) {
            throw MinervaCompilerException(
                message = "Failed to prepare Minerva compiler input: ${exception.message ?: exception.toString()}",
                code = "minerva.compiler.input_write_failed",
                prerequisite = true,
                remediation = "Ensure outputDir is writable and has enough space.",
                details = mapOf("modelPath" to modelPath.toString())
            )
        }

        val command = buildCommand(options, scriptPath, modelPath, generatedDir)
        val commandSummary = summarizeCommand(command)
        context.info(
            stage = GraphExportStage.PACKAGING,
            code = "minerva.compiler.started",
            message = "Invoking libminerva compiler.",
            details = mapOf(
                "command" to commandSummary,
                "model" to modelPath.toString(),
                "outputDir" to generatedDir.toString()
            )
        )

        val processResult = runProcess(command, projectDir, commandSummary)
        if (processResult.exitCode != 0) {
            throw MinervaCompilerException(
                message = "Minerva compiler failed with exit code ${processResult.exitCode}.",
                code = "minerva.compiler.process_failed",
                stdout = processResult.stdout,
                stderr = processResult.stderr,
                exitCode = processResult.exitCode,
                commandSummary = commandSummary,
                remediation = "Inspect compiler stdout/stderr and verify target, quantization, calibration, and key configuration.",
                details = mapOf("outputDir" to generatedDir.toString())
            )
        }

        val weightsC = generatedDir.resolve("weights.c")
        val weightsH = generatedDir.resolve("weights.h")
        requireCompilerOutput(weightsC, "weights.c", commandSummary, processResult)
        requireCompilerOutput(weightsH, "weights.h", commandSummary, processResult)
        val debugWeights = generatedDir.resolve("weights_debug.npz").takeIf { Files.exists(it) }
        val output = MinervaCompilerOutput(
            outputDir = generatedDir.toString(),
            weightsCPath = weightsC.toString(),
            weightsHPath = weightsH.toString(),
            debugWeightsPath = debugWeights?.toString(),
            commandSummary = commandSummary,
            stdout = processResult.stdout,
            stderr = processResult.stderr,
            exitCode = processResult.exitCode,
            metadata = mapOf(
                "target" to options.target.compilerId,
                "quantization" to options.quantization.compilerId,
                "model" to modelPath.toString()
            )
        )
        context.addArtifact(
            GraphExportArtifact(
                path = output.weightsCPath,
                role = GraphExportArtifactRole.SOURCE,
                description = "Minerva compiler weights source"
            )
        )
        context.addArtifact(
            GraphExportArtifact(
                path = output.weightsHPath,
                role = GraphExportArtifactRole.HEADER,
                description = "Minerva compiler weights header"
            )
        )
        context.info(
            stage = GraphExportStage.PACKAGING,
            code = "minerva.compiler.completed",
            message = "libminerva compiler completed successfully.",
            details = mapOf(
                "weightsC" to output.weightsCPath,
                "weightsH" to output.weightsHPath,
                "exitCode" to output.exitCode.toString()
            )
        )
        return output
    }

    private fun buildCommand(
        options: MinervaExportOptions,
        scriptPath: Path,
        modelPath: Path,
        generatedDir: Path
    ): List<String> {
        val command = mutableListOf(
            options.pythonExecutable,
            scriptPath.toAbsolutePath().normalize().toString(),
            "--model",
            modelPath.toAbsolutePath().normalize().toString(),
            "--out-dir",
            generatedDir.toAbsolutePath().normalize().toString(),
            "--target",
            options.target.compilerId,
            "--quantization",
            options.quantization.compilerId
        )
        options.runtimeRoot?.let {
            command += listOf("--runtime-root", Paths.get(it).toAbsolutePath().normalize().toString())
        }
        options.keyFile?.let {
            command += listOf("--key-file", Paths.get(it).toAbsolutePath().normalize().toString())
        }
        options.calibrationNpz?.let {
            command += listOf("--calibration", Paths.get(it).toAbsolutePath().normalize().toString())
        }
        if (options.dumpWeights) command += "--dump-weights"
        return command
    }

    private fun runProcess(
        command: List<String>,
        workingDir: Path,
        commandSummary: String
    ): ProcessResult {
        try {
            Files.createDirectories(workingDir)
            val process = ProcessBuilder(command)
                .directory(workingDir.toFile())
                .start()
            val stdout = StreamCollector(process.inputStream).also { it.start() }
            val stderr = StreamCollector(process.errorStream).also { it.start() }
            val exitCode = process.waitFor()
            stdout.join()
            stderr.join()
            return ProcessResult(
                exitCode = exitCode,
                stdout = stdout.text(),
                stderr = stderr.text()
            )
        } catch (exception: IOException) {
            throw MinervaCompilerException(
                message = "Failed to start Minerva compiler process: ${exception.message ?: exception.toString()}",
                code = "minerva.compiler.process_start_failed",
                prerequisite = true,
                commandSummary = commandSummary,
                remediation = "Verify pythonExecutable and compilerScript are executable in this environment.",
                details = mapOf("workingDir" to workingDir.toString())
            )
        } catch (exception: InterruptedException) {
            Thread.currentThread().interrupt()
            throw MinervaCompilerException(
                message = "Minerva compiler process was interrupted.",
                code = "minerva.compiler.process_interrupted",
                commandSummary = commandSummary,
                remediation = "Retry the export when the build process can run to completion.",
                details = mapOf("workingDir" to workingDir.toString())
            )
        }
    }

    private fun requireCompilerOutput(
        path: Path,
        name: String,
        commandSummary: String,
        processResult: ProcessResult
    ) {
        if (!Files.isRegularFile(path)) {
            throw MinervaCompilerException(
                message = "Minerva compiler did not produce required output '$name'.",
                code = "minerva.compiler.output_missing",
                stdout = processResult.stdout,
                stderr = processResult.stderr,
                exitCode = processResult.exitCode,
                commandSummary = commandSummary,
                remediation = "Verify the libminerva compiler version and output directory contract.",
                details = mapOf("missingPath" to path.toString())
            )
        }
    }

    private fun requireRegularFile(
        field: String,
        value: String,
        code: String,
        remediation: String
    ): Path {
        val path = Paths.get(value)
        if (!Files.isRegularFile(path)) {
            prerequisiteFailure(
                code = code,
                message = "Minerva compiler prerequisite '$field' does not exist or is not a file: $value",
                remediation = remediation,
                details = mapOf(field to value)
            )
        }
        return path
    }

    private fun requireDirectory(
        field: String,
        value: String,
        code: String,
        remediation: String
    ): Path {
        val path = Paths.get(value)
        if (!Files.isDirectory(path)) {
            prerequisiteFailure(
                code = code,
                message = "Minerva compiler prerequisite '$field' does not exist or is not a directory: $value",
                remediation = remediation,
                details = mapOf(field to value)
            )
        }
        return path
    }

    private fun prerequisiteFailure(
        code: String,
        message: String,
        remediation: String,
        details: Map<String, String> = emptyMap()
    ): Nothing {
        throw MinervaCompilerException(
            message = message,
            code = code,
            prerequisite = true,
            remediation = remediation,
            details = details
        )
    }

    private fun summarizeCommand(command: List<String>): String {
        return command.mapIndexed { index, value ->
            val previous = command.getOrNull(index - 1)
            when (previous) {
                "--key-file" -> "<key-file>"
                else -> value
            }
        }.joinToString(" ")
    }
}

/**
 * JVM packager that writes a host/firmware-oriented Minerva project layout.
 */
public class JvmMinervaProjectPackager @kotlin.jvm.JvmOverloads constructor(
    override val backendName: String = MinervaExportBackend.backendName
) : MinervaProjectPackager {

    override fun packageProject(
        request: MinervaProjectPackageRequest,
        context: GraphExportContext
    ): MinervaExportBundle {
        val options = request.options
        val projectDir = Paths.get(options.outputDir).resolve(options.projectName).normalize()
        val generatedDir = projectDir.resolve("generated")
        val includeDir = projectDir.resolve("include")
        val hostDir = projectDir.resolve("host")
        val firmwareDir = projectDir.resolve("firmware")
        context.info(
            stage = GraphExportStage.PACKAGING,
            code = "minerva.packaging.started",
            message = "Packaging Minerva project outputs.",
            details = mapOf("projectDir" to projectDir.toString())
        )

        try {
            listOf(projectDir, generatedDir, includeDir, hostDir, firmwareDir).forEach(Files::createDirectories)
            val modelPath = generatedDir.resolve(request.npzModel.logicalPath.substringAfterLast('/'))
            Files.write(modelPath, request.npzModel.bytes)
            val weightsC = copyCompilerOutput(
                source = Paths.get(request.compilerOutput.weightsCPath),
                target = generatedDir.resolve("weights.c"),
                logicalName = "weights.c"
            )
            val weightsH = copyCompilerOutput(
                source = Paths.get(request.compilerOutput.weightsHPath),
                target = includeDir.resolve("weights.h"),
                logicalName = "weights.h"
            )
            val debugWeights = request.compilerOutput.debugWeightsPath?.let { debugWeightsPath ->
                val source = Paths.get(debugWeightsPath)
                val fileName = source.fileName?.toString() ?: "weights_debug.npz"
                copyCompilerOutput(
                    source = source,
                    target = generatedDir.resolve(fileName),
                    logicalName = fileName
                )
            }
            val secretsExample = includeDir.resolve("secrets.example.h")
            Files.writeString(secretsExample, secretsExampleHeader(options))
            val generatedPaths = mutableListOf(modelPath, weightsC, weightsH, secretsExample)
            debugWeights?.let(generatedPaths::add)
            if (options.generateHostHarness) {
                val hostCmake = hostDir.resolve("CMakeLists.txt")
                val hostMain = hostDir.resolve("main.c")
                Files.writeString(hostCmake, hostCmake(options))
                Files.writeString(hostMain, hostMain(request))
                generatedPaths.add(hostCmake)
                generatedPaths.add(hostMain)
            }
            if (options.generateFirmwareExample) {
                val firmwareMain = firmwareDir.resolve("main.c")
                Files.writeString(firmwareMain, firmwareMain(request))
                generatedPaths.add(firmwareMain)
            }

            val manifestPath = projectDir.resolve("manifest.json")
            val generatedRelative = generatedPaths.map { relativePath(projectDir, it) }
            Files.writeString(
                manifestPath,
                manifestJson(
                    request = request,
                    generatedFiles = generatedRelative,
                    manifestPath = relativePath(projectDir, manifestPath)
                )
            )
            val allRelative = generatedRelative + relativePath(projectDir, manifestPath)
            recordArtifacts(context, projectDir, manifestPath, generatedPaths)
            context.info(
                stage = GraphExportStage.PACKAGING,
                code = "minerva.packaging.completed",
                message = "Packaged Minerva project outputs.",
                details = mapOf(
                    "projectDir" to projectDir.toString(),
                    "files" to allRelative.size.toString(),
                    "manifest" to relativePath(projectDir, manifestPath)
                )
            )
            return MinervaExportBundle(
                projectName = options.projectName,
                outputDir = projectDir.toString(),
                target = options.target,
                quantization = options.quantization,
                generatedFiles = allRelative,
                manifestPath = relativePath(projectDir, manifestPath),
                compilerOutput = request.compilerOutput
            )
        } catch (exception: MinervaPackagingException) {
            throw exception
        } catch (exception: IOException) {
            throw MinervaPackagingException(
                message = "Failed to package Minerva project: ${exception.message ?: exception.toString()}",
                code = "minerva.packaging.io_failed",
                remediation = "Ensure outputDir is writable and compiler outputs are readable.",
                details = mapOf("projectDir" to projectDir.toString())
            )
        }
    }

    private fun copyCompilerOutput(source: Path, target: Path, logicalName: String): Path {
        if (!Files.isRegularFile(source)) {
            throw MinervaPackagingException(
                message = "Cannot package missing compiler output '$logicalName'.",
                code = "minerva.packaging.compiler_output_missing",
                remediation = "Run the compiler adapter successfully before packaging.",
                details = mapOf("missingPath" to source.toString())
            )
        }
        Files.createDirectories(target.parent)
        if (source.toAbsolutePath().normalize() != target.toAbsolutePath().normalize()) {
            Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING)
        }
        return target
    }

    private fun recordArtifacts(
        context: GraphExportContext,
        projectDir: Path,
        manifestPath: Path,
        generatedPaths: List<Path>
    ) {
        context.addArtifact(
            GraphExportArtifact(
                path = projectDir.toString(),
                role = GraphExportArtifactRole.PROJECT_DIRECTORY,
                description = "Minerva packaged project directory"
            )
        )
        context.addArtifact(
            GraphExportArtifact(
                path = manifestPath.toString(),
                role = GraphExportArtifactRole.MANIFEST,
                description = "Minerva export manifest"
            )
        )
        generatedPaths.forEach { path ->
            val role = when {
                path.fileName.toString().endsWith(".h") -> GraphExportArtifactRole.HEADER
                path.fileName.toString().endsWith(".npz") -> GraphExportArtifactRole.INTERMEDIATE
                else -> GraphExportArtifactRole.SOURCE
            }
            context.addArtifact(
                GraphExportArtifact(
                    path = path.toString(),
                    role = role,
                    description = "Minerva packaged file ${relativePath(projectDir, path)}",
                    sensitive = false
                )
            )
        }
    }

    private fun manifestJson(
        request: MinervaProjectPackageRequest,
        generatedFiles: List<String>,
        manifestPath: String
    ): String {
        val options = request.options
        val values = mapOf(
            "projectName" to jsonString(options.projectName),
            "skainetVersion" to jsonString(options.metadata["skainetVersion"] ?: "unknown"),
            "libminerva" to jsonString(options.runtimeRoot ?: "unspecified"),
            "target" to jsonString(options.target.compilerId),
            "quantization" to jsonString(options.quantization.compilerId),
            "compilerCommand" to jsonString(request.compilerOutput.commandSummary),
            "compilerExitCode" to request.compilerOutput.exitCode.toString(),
            "npzSchemaVersion" to request.npzModel.schemaVersion.toString(),
            "layers" to request.intermediate.layerCount.toString(),
            "hostHarness" to options.generateHostHarness.toString(),
            "firmwareExample" to options.generateFirmwareExample.toString(),
            "manifestPath" to jsonString(manifestPath),
            "generatedFiles" to generatedFiles.joinToString(prefix = "[", postfix = "]") { jsonString(it) }
        )
        return values.entries.joinToString(prefix = "{\n", postfix = "\n}\n", separator = ",\n") { (key, value) ->
            "  ${jsonString(key)}: $value"
        }
    }

    private fun secretsExampleHeader(options: MinervaExportOptions): String {
        return """
            |#pragma once
            |
            |/*
            | * Example-only Minerva secret configuration for ${options.projectName}.
            | * Replace these placeholders in a private, untracked file.
            | */
            |#define MINERVA_DEVICE_KEY_HEX "replace-with-device-key"
            |#define MINERVA_KEY_ID "replace-with-key-id"
            |
        """.trimMargin()
    }

    private fun hostCmake(options: MinervaExportOptions): String {
        return """
            |cmake_minimum_required(VERSION 3.20)
            |project(${options.projectName}_host C)
            |
            |add_executable(${options.projectName}_host main.c ../generated/weights.c)
            |target_include_directories(${options.projectName}_host PRIVATE ../include)
            |
        """.trimMargin()
    }

    private fun hostMain(request: MinervaProjectPackageRequest): String {
        val inputCount = request.intermediate.input.elementCount
        val outputCount = request.intermediate.output.elementCount
        return """
            |#include <stdint.h>
            |#include <stdio.h>
            |#include "weights.h"
            |
            |int main(void) {
            |    float input[$inputCount] = {0};
            |    float output[$outputCount] = {0};
            |
            |    /* Link this harness with libminerva and call the runtime inference entry point here. */
            |    (void)input;
            |    (void)output;
            |    puts("Minerva host harness packaged successfully.");
            |    return 0;
            |}
            |
        """.trimMargin()
    }

    private fun firmwareMain(request: MinervaProjectPackageRequest): String {
        val inputCount = request.intermediate.input.elementCount
        val outputCount = request.intermediate.output.elementCount
        return """
            |#include <stdint.h>
            |#include "weights.h"
            |#include "secrets.example.h"
            |
            |void setup(void) {
            |    /* Initialize Minerva runtime and seed the PRNG before inference. */
            |}
            |
            |void loop(void) {
            |    float input[$inputCount] = {0};
            |    float output[$outputCount] = {0};
            |
            |    /* Call the libminerva inference function with input and output buffers here. */
            |    (void)input;
            |    (void)output;
            |}
            |
        """.trimMargin()
    }

    private fun relativePath(root: Path, path: Path): String {
        return root.toAbsolutePath().normalize()
            .relativize(path.toAbsolutePath().normalize())
            .toString()
            .replace('\\', '/')
    }

    private fun jsonString(value: String): String {
        val escaped = buildString {
            value.forEach { char ->
                when (char) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(char)
                }
            }
        }
        return "\"$escaped\""
    }
}

private data class ProcessResult(
    val exitCode: Int,
    val stdout: String,
    val stderr: String
)

private class StreamCollector(private val stream: InputStream) : Thread("minerva-process-stream") {
    private val output = ByteArrayOutputStream()

    override fun run() {
        stream.use { input -> input.copyTo(output) }
    }

    fun text(): String = output.toString(Charsets.UTF_8.name())
}
