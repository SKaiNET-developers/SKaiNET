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

private const val HOST_REFERENCE_INPUT_FILE = "reference-input.txt"
private const val HOST_REFERENCE_OUTPUT_FILE = "reference-output.txt"
private const val HOST_OBSERVED_OUTPUT_FILE = "observed-output.txt"

public actual object MinervaPlatformExportDefaults {
    public actual fun compilerAdapter(): MinervaCompilerAdapter = PythonMinervaCompilerAdapter()

    public actual fun projectPackager(): MinervaProjectPackager = JvmMinervaProjectPackager()

    public actual fun hostVerifier(): MinervaHostVerifier = JvmMinervaHostVerifier()
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
            val referenceInput = MinervaReferenceEvaluator.referenceInput(request.intermediate.input)
            val referenceOutput = try {
                MinervaReferenceEvaluator.evaluate(request.intermediate, referenceInput)
            } catch (exception: RuntimeException) {
                throw MinervaPackagingException(
                    message = "Failed to create Minerva reference output fixture: ${exception.message ?: exception.toString()}",
                    code = "minerva.packaging.reference_fixture_failed",
                    remediation = "Ensure lowered Minerva weights and biases contain numeric initializer values.",
                    details = mapOf("projectName" to options.projectName)
                )
            }
            val referenceInputPath = hostDir.resolve(HOST_REFERENCE_INPUT_FILE)
            val referenceOutputPath = hostDir.resolve(HOST_REFERENCE_OUTPUT_FILE)
            Files.writeString(referenceInputPath, floatValuesText(referenceInput))
            Files.writeString(referenceOutputPath, floatValuesText(referenceOutput))
            val generatedPaths = mutableListOf(
                modelPath,
                weightsC,
                weightsH,
                secretsExample,
                referenceInputPath,
                referenceOutputPath
            )
            debugWeights?.let(generatedPaths::add)
            if (options.generateHostHarness) {
                val hostCmake = hostDir.resolve("CMakeLists.txt")
                val hostMain = hostDir.resolve("main.c")
                val hostAdapterExample = hostDir.resolve("runtime_adapter.example.c")
                Files.writeString(hostCmake, hostCmake(options))
                Files.writeString(hostMain, hostMain(request))
                Files.writeString(hostAdapterExample, hostRuntimeAdapterExample())
                generatedPaths.add(hostCmake)
                generatedPaths.add(hostMain)
                generatedPaths.add(hostAdapterExample)
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
                path.fileName.toString().startsWith("reference-") -> GraphExportArtifactRole.TEST_REPORT
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
            "referenceInputPath" to jsonString("host/$HOST_REFERENCE_INPUT_FILE"),
            "referenceOutputPath" to jsonString("host/$HOST_REFERENCE_OUTPUT_FILE"),
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
        val runtimeRoot = cmakeString(options.runtimeRoot ?: "")
        val adapterSource = cmakeString(options.metadata[MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE] ?: "")
        val includeDirs = cmakeString(options.metadata[MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS] ?: "")
        val libraryDirs = cmakeString(options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS] ?: "")
        val libraries = cmakeString(options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARIES] ?: "")
        return """
            |cmake_minimum_required(VERSION 3.20)
            |project(${options.projectName}_host C)
            |
            |set(MINERVA_RUNTIME_ROOT "$runtimeRoot" CACHE PATH "libminerva checkout or install root")
            |set(MINERVA_HOST_ADAPTER_SOURCE "$adapterSource" CACHE FILEPATH "C source file implementing minerva_run_inference")
            |set(MINERVA_HOST_INCLUDE_DIRS "$includeDirs" CACHE STRING "Additional semicolon-separated Minerva host include directories")
            |set(MINERVA_HOST_LIBRARY_DIRS "$libraryDirs" CACHE STRING "Additional semicolon-separated Minerva host library directories")
            |set(MINERVA_HOST_LIBRARIES "$libraries" CACHE STRING "Additional semicolon-separated Minerva host libraries")
            |
            |set(MINERVA_HOST_SOURCES main.c ../generated/weights.c)
            |if(MINERVA_HOST_ADAPTER_SOURCE)
            |  list(APPEND MINERVA_HOST_SOURCES "${'$'}{MINERVA_HOST_ADAPTER_SOURCE}")
            |endif()
            |
            |add_executable(${options.projectName}_host ${'$'}{MINERVA_HOST_SOURCES})
            |target_include_directories(${options.projectName}_host PRIVATE ../include)
            |if(MINERVA_RUNTIME_ROOT)
            |  target_include_directories(${options.projectName}_host PRIVATE "${'$'}{MINERVA_RUNTIME_ROOT}/include")
            |endif()
            |if(MINERVA_HOST_INCLUDE_DIRS)
            |  target_include_directories(${options.projectName}_host PRIVATE ${'$'}{MINERVA_HOST_INCLUDE_DIRS})
            |endif()
            |if(MINERVA_HOST_LIBRARY_DIRS)
            |  target_link_directories(${options.projectName}_host PRIVATE ${'$'}{MINERVA_HOST_LIBRARY_DIRS})
            |endif()
            |if(MINERVA_HOST_LIBRARIES)
            |  target_link_libraries(${options.projectName}_host PRIVATE ${'$'}{MINERVA_HOST_LIBRARIES})
            |endif()
            |if(MINERVA_HOST_ADAPTER_SOURCE)
            |  target_compile_definitions(${options.projectName}_host PRIVATE MINERVA_HOST_RUNTIME_ENABLED=1)
            |endif()
            |
            |include(CTest)
            |if(BUILD_TESTING)
            |  add_test(NAME minerva_host_smoke COMMAND ${options.projectName}_host)
            |  set_tests_properties(minerva_host_smoke PROPERTIES WORKING_DIRECTORY "${'$'}{CMAKE_CURRENT_SOURCE_DIR}")
            |endif()
            |
        """.trimMargin()
    }

    private fun hostMain(request: MinervaProjectPackageRequest): String {
        val inputCount = request.intermediate.input.elementCount
        val outputCount = request.intermediate.output.elementCount
        val referenceInput = MinervaReferenceEvaluator.referenceInput(request.intermediate.input)
        return """
            |#include <stdint.h>
            |#include <stdio.h>
            |#include "weights.h"
            |
            |#define MINERVA_OBSERVED_OUTPUT_PATH "$HOST_OBSERVED_OUTPUT_FILE"
            |
            |#ifdef MINERVA_HOST_RUNTIME_ENABLED
            |extern int minerva_run_inference(const float *input, int input_count, float *output, int output_count);
            |
            |static int write_observed_output(const float *output, int output_count) {
            |    FILE *file = fopen(MINERVA_OBSERVED_OUTPUT_PATH, "w");
            |    if (file == NULL) {
            |        perror("failed to open " MINERVA_OBSERVED_OUTPUT_PATH);
            |        return 1;
            |    }
            |    for (int i = 0; i < output_count; ++i) {
            |        if (fprintf(file, "%s%.9g", i == 0 ? "" : "\n", (double)output[i]) < 0) {
            |            fclose(file);
            |            return 1;
            |        }
            |    }
            |    if (fprintf(file, "\n") < 0) {
            |        fclose(file);
            |        return 1;
            |    }
            |    return fclose(file) == 0 ? 0 : 1;
            |}
            |#endif
            |
            |int main(void) {
            |    float input[$inputCount] = {${cFloatInitializer(referenceInput)}};
            |    float output[$outputCount] = {0};
            |
            |#ifdef MINERVA_HOST_RUNTIME_ENABLED
            |    int status = minerva_run_inference(input, $inputCount, output, $outputCount);
            |    if (status != 0) {
            |        fprintf(stderr, "minerva_run_inference failed with status %d\n", status);
            |        return status;
            |    }
            |    if (write_observed_output(output, $outputCount) != 0) {
            |        fprintf(stderr, "failed to write " MINERVA_OBSERVED_OUTPUT_PATH "\n");
            |        return 2;
            |    }
            |    puts("Minerva host harness wrote " MINERVA_OBSERVED_OUTPUT_PATH ".");
            |#else
            |    (void)input;
            |    (void)output;
            |    puts("Minerva host harness packaged successfully.");
            |    puts("Reference input: $HOST_REFERENCE_INPUT_FILE");
            |    puts("Write observed output to: $HOST_OBSERVED_OUTPUT_FILE");
            |#endif
            |    return 0;
            |}
            |
        """.trimMargin()
    }

    private fun hostRuntimeAdapterExample(): String {
        return """
            |/*
            | * Copy this file outside the generated bundle and wire it to your pinned
            | * libminerva runtime. Then configure CMake with:
            | *
            | *   -DMINERVA_HOST_ADAPTER_SOURCE=/path/to/minerva_runtime_adapter.c
            | *
            | * The generated host harness calls this stable shim so SKaiNET does not
            | * need to hard-code libminerva runtime entry point names.
            | */
            |#include "weights.h"
            |
            |int minerva_run_inference(const float *input, int input_count, float *output, int output_count) {
            |    (void)input;
            |    (void)input_count;
            |    (void)output;
            |    (void)output_count;
            |    return 1;
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

    private fun floatValuesText(values: List<Float>): String {
        return values.joinToString(separator = "\n", postfix = "\n") { it.toString() }
    }

    private fun cFloatInitializer(values: List<Float>): String {
        return values.joinToString(separator = ", ") { "${it}f" }
    }

    private fun cmakeString(value: String): String {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
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

/**
 * JVM host verifier for packaged Minerva projects.
 */
public class JvmMinervaHostVerifier @kotlin.jvm.JvmOverloads constructor(
    override val backendName: String = MinervaExportBackend.backendName
) : MinervaHostVerifier {

    override fun verify(
        request: MinervaHostVerificationRequest,
        context: GraphExportContext
    ): MinervaHostVerification {
        val options = request.options
        val projectDir = Paths.get(request.bundle.outputDir).normalize()
        val tolerance = options.hostVerificationTolerance
        context.info(
            stage = GraphExportStage.VERIFICATION,
            code = "minerva.host_verification.started",
            message = "Verifying packaged Minerva host project.",
            details = mapOf(
                "projectDir" to projectDir.toString(),
                "tolerance" to tolerance.toString()
            )
        )

        try {
            structuralFailure(request, projectDir)?.let { return it }
        } catch (exception: IOException) {
            return failed(
                code = "minerva.host_verification.package_read_failed",
                message = "Unable to read packaged Minerva project files during host verification.",
                tolerance = tolerance,
                remediation = "Ensure the packaged project directory is readable and was not modified during verification.",
                details = mapOf("reason" to (exception.message ?: exception.toString()))
            )
        }
        val referenceInput = try {
            readFloatOutput(projectDir.resolve("host/$HOST_REFERENCE_INPUT_FILE"))
        } catch (exception: IllegalArgumentException) {
            return failed(
                code = "minerva.host_verification.reference_input_invalid",
                message = "Packaged Minerva reference input fixture could not be parsed.",
                tolerance = tolerance,
                remediation = "Regenerate the Minerva package so host/reference-input.txt contains finite float values.",
                details = mapOf("reason" to (exception.message ?: exception.toString()))
            )
        }
        if (referenceInput.size != request.intermediate.input.elementCount) {
            return failed(
                code = "minerva.host_verification.reference_input_shape_mismatch",
                message = "Packaged Minerva reference input length does not match the lowered model input tensor.",
                tolerance = tolerance,
                remediation = "Regenerate the Minerva package from the current lowered graph.",
                details = mapOf(
                    "expected" to request.intermediate.input.elementCount.toString(),
                    "observed" to referenceInput.size.toString()
                )
            )
        }

        val expectedOutput = try {
            MinervaReferenceEvaluator.evaluate(request.intermediate, referenceInput)
        } catch (exception: RuntimeException) {
            return failed(
                code = "minerva.host_verification.reference_unavailable",
                message = "Unable to compute SKaiNET reference output for Minerva parity verification.",
                tolerance = tolerance,
                remediation = "Ensure lowered Minerva weights and biases contain numeric initializer values.",
                details = mapOf("reason" to (exception.message ?: exception.toString()))
            )
        }
        validateReferenceOutputFixture(projectDir, expectedOutput, tolerance)?.let { return it }

        var hostBuildStatus = MinervaHostVerificationStatus.SKIPPED
        var hostRunStatus = MinervaHostVerificationStatus.SKIPPED
        if (metadataFlag(options, MinervaHostVerificationMetadata.RUN_CMAKE_BUILD)) {
            val buildFailure = runCmakeBuild(projectDir, options, context, tolerance, expectedOutput)
            if (buildFailure != null) return buildFailure
            hostBuildStatus = MinervaHostVerificationStatus.PASSED
            if (metadataFlag(options, MinervaHostVerificationMetadata.RUN_CTEST)) {
                val testFailure = runCTest(projectDir, options, context, tolerance, expectedOutput)
                if (testFailure != null) return testFailure
                hostRunStatus = MinervaHostVerificationStatus.PASSED
            }
        }

        val hostOutputPath = options.metadata[MinervaHostVerificationMetadata.HOST_OUTPUT_PATH]
            ?: "host/$HOST_OBSERVED_OUTPUT_FILE".takeIf { Files.isRegularFile(projectDir.resolve(it)) }
        val observedOutput = if (hostOutputPath != null) {
            val outputPath = resolveProjectPath(projectDir, hostOutputPath)
            try {
                readFloatOutput(outputPath)
            } catch (exception: IllegalArgumentException) {
                return failed(
                    code = "minerva.host_verification.host_output_invalid",
                    message = "Configured Minerva host output could not be parsed.",
                    tolerance = tolerance,
                    expectedOutput = expectedOutput,
                    remediation = "Write host output as whitespace- or comma-separated finite float values.",
                    details = mapOf(
                        "hostOutputPath" to outputPath.toString(),
                        "reason" to (exception.message ?: exception.toString())
                    )
                )
            }
        } else {
            emptyList()
        }

        val parityStatus: MinervaHostVerificationStatus
        val maxAbsoluteError: Float?
        if (observedOutput.isNotEmpty()) {
            if (expectedOutput.size != observedOutput.size) {
                return failed(
                    code = "minerva.host_verification.parity_shape_mismatch",
                    message = "Minerva host output length does not match the SKaiNET reference output length.",
                    tolerance = tolerance,
                    expectedOutput = expectedOutput,
                    observedOutput = observedOutput,
                    remediation = "Regenerate the host output from the same packaged project and reference input.",
                    details = mapOf(
                        "expected" to expectedOutput.size.toString(),
                        "observed" to observedOutput.size.toString()
                    )
                )
            }
            maxAbsoluteError = MinervaReferenceEvaluator.maxAbsoluteError(expectedOutput, observedOutput)
            if (maxAbsoluteError > tolerance) {
                return failed(
                    code = "minerva.host_verification.parity_failed",
                    message = "Minerva host output differs from the SKaiNET reference output.",
                    tolerance = tolerance,
                    maxAbsoluteError = maxAbsoluteError,
                    expectedOutput = expectedOutput,
                    observedOutput = observedOutput,
                    remediation = "Inspect compiler inputs, generated weights, quantization settings, and host runtime configuration.",
                    details = mapOf("maxAbsoluteError" to maxAbsoluteError.toString())
                )
            }
            parityStatus = MinervaHostVerificationStatus.PASSED
            hostRunStatus = MinervaHostVerificationStatus.PASSED
        } else {
            parityStatus = MinervaHostVerificationStatus.SKIPPED
            maxAbsoluteError = null
        }

        val verification = MinervaHostVerification(
            status = MinervaHostVerificationStatus.PASSED,
            code = "minerva.host_verification.passed",
            message = "Minerva host verification completed.",
            hostBuildStatus = hostBuildStatus,
            hostRunStatus = hostRunStatus,
            parityStatus = parityStatus,
            tolerance = tolerance,
            maxAbsoluteError = maxAbsoluteError,
            expectedOutput = expectedOutput,
            observedOutput = observedOutput,
            details = mapOf(
                "projectDir" to projectDir.toString(),
                "referenceInputPath" to "host/$HOST_REFERENCE_INPUT_FILE",
                "referenceOutputPath" to "host/$HOST_REFERENCE_OUTPUT_FILE",
                "referenceOutputValues" to expectedOutput.size.toString()
            )
        )
        context.info(
            stage = GraphExportStage.VERIFICATION,
            code = verification.code,
            message = verification.message,
            details = mapOf(
                "hostBuildStatus" to verification.hostBuildStatus.name,
                "hostRunStatus" to verification.hostRunStatus.name,
                "parityStatus" to verification.parityStatus.name,
                "maxAbsoluteError" to (verification.maxAbsoluteError?.toString() ?: "n/a")
            )
        )
        return verification
    }

    private fun validateReferenceOutputFixture(
        projectDir: Path,
        expectedOutput: List<Float>,
        tolerance: Float
    ): MinervaHostVerification? {
        val outputPath = projectDir.resolve("host/$HOST_REFERENCE_OUTPUT_FILE")
        val fixtureOutput = try {
            readFloatOutput(outputPath)
        } catch (exception: IllegalArgumentException) {
            return failed(
                code = "minerva.host_verification.reference_output_invalid",
                message = "Packaged Minerva reference output fixture could not be parsed.",
                tolerance = tolerance,
                expectedOutput = expectedOutput,
                remediation = "Regenerate the Minerva package so host/reference-output.txt contains finite float values.",
                details = mapOf(
                    "referenceOutputPath" to outputPath.toString(),
                    "reason" to (exception.message ?: exception.toString())
                )
            )
        }
        if (fixtureOutput.size != expectedOutput.size) {
            return failed(
                code = "minerva.host_verification.reference_output_shape_mismatch",
                message = "Packaged Minerva reference output length does not match the SKaiNET reference output length.",
                tolerance = tolerance,
                expectedOutput = expectedOutput,
                observedOutput = fixtureOutput,
                remediation = "Regenerate the Minerva package from the current lowered graph.",
                details = mapOf(
                    "expected" to expectedOutput.size.toString(),
                    "observed" to fixtureOutput.size.toString()
                )
            )
        }
        val maxAbsoluteError = MinervaReferenceEvaluator.maxAbsoluteError(expectedOutput, fixtureOutput)
        if (maxAbsoluteError > tolerance) {
            return failed(
                code = "minerva.host_verification.reference_output_mismatch",
                message = "Packaged Minerva reference output differs from the SKaiNET reference evaluator.",
                tolerance = tolerance,
                maxAbsoluteError = maxAbsoluteError,
                expectedOutput = expectedOutput,
                observedOutput = fixtureOutput,
                remediation = "Regenerate the Minerva package and inspect lowered layer metadata for stale fixture files.",
                details = mapOf("maxAbsoluteError" to maxAbsoluteError.toString())
            )
        }
        return null
    }

    private fun structuralFailure(
        request: MinervaHostVerificationRequest,
        projectDir: Path
    ): MinervaHostVerification? {
        val tolerance = request.options.hostVerificationTolerance
        val requiredFiles = buildList {
            add(projectDir.resolve("manifest.json"))
            add(projectDir.resolve("generated").resolve(request.npzModel.logicalPath.substringAfterLast('/')))
            add(projectDir.resolve("generated/weights.c"))
            add(projectDir.resolve("include/weights.h"))
            add(projectDir.resolve("include/secrets.example.h"))
            add(projectDir.resolve("host/$HOST_REFERENCE_INPUT_FILE"))
            add(projectDir.resolve("host/$HOST_REFERENCE_OUTPUT_FILE"))
            if (request.options.generateHostHarness) {
                add(projectDir.resolve("host/CMakeLists.txt"))
                add(projectDir.resolve("host/main.c"))
                add(projectDir.resolve("host/runtime_adapter.example.c"))
            }
            if (request.options.generateFirmwareExample) {
                add(projectDir.resolve("firmware/main.c"))
            }
        }
        requiredFiles.firstOrNull { !Files.isRegularFile(it) }?.let { missing ->
            return failed(
                code = "minerva.host_verification.required_file_missing",
                message = "Packaged Minerva project is missing a required generated file.",
                tolerance = tolerance,
                remediation = "Re-run Minerva packaging and verify the generated project layout.",
                details = mapOf("missingPath" to missing.toString())
            )
        }
        listOf(projectDir.resolve("generated/weights.c"), projectDir.resolve("include/weights.h"))
            .firstOrNull { Files.size(it) == 0L }
            ?.let { empty ->
                return failed(
                    code = "minerva.host_verification.empty_generated_file",
                    message = "Packaged Minerva project contains an empty compiler output.",
                    tolerance = tolerance,
                    remediation = "Inspect the libminerva compiler invocation and generated weights.",
                    details = mapOf("emptyPath" to empty.toString())
                )
            }
        val packagedModel = projectDir.resolve("generated").resolve(request.npzModel.logicalPath.substringAfterLast('/'))
        if (!Files.readAllBytes(packagedModel).contentEquals(request.npzModel.bytes)) {
            return failed(
                code = "minerva.host_verification.model_tampered",
                message = "Packaged Minerva model bytes differ from the NPZ compiler input produced by SKaiNET.",
                tolerance = tolerance,
                remediation = "Recreate the package from the original export result before running host verification.",
                details = mapOf("modelPath" to packagedModel.toString())
            )
        }
        secretLeakFailure(request, projectDir)?.let { return it }
        return null
    }

    private fun secretLeakFailure(
        request: MinervaHostVerificationRequest,
        projectDir: Path
    ): MinervaHostVerification? {
        val keyPath = request.options.keyFile?.let(Paths::get) ?: return null
        if (!Files.isRegularFile(keyPath) || Files.size(keyPath) == 0L || Files.size(keyPath) > 4096L) return null
        val keyMaterial = Files.readString(keyPath).trim()
        if (keyMaterial.isBlank()) return null
        val secretsExample = projectDir.resolve("include/secrets.example.h")
        val template = Files.readString(secretsExample)
        if (!template.contains(keyMaterial)) return null
        return failed(
            code = "minerva.host_verification.secret_leak",
            message = "Generated Minerva secret template contains real key material.",
            tolerance = request.options.hostVerificationTolerance,
            remediation = "Remove real secrets from generated artifacts and regenerate secrets.example.h with placeholders.",
            details = mapOf("secretsExample" to secretsExample.toString())
        )
    }

    private fun runCmakeBuild(
        projectDir: Path,
        options: MinervaExportOptions,
        context: GraphExportContext,
        tolerance: Float,
        expectedOutput: List<Float>
    ): MinervaHostVerification? {
        val hostDir = projectDir.resolve("host")
        val buildDir = hostDir.resolve("build")
        Files.createDirectories(buildDir)
        val cmake = options.metadata[MinervaHostVerificationMetadata.CMAKE_EXECUTABLE] ?: "cmake"
        val configure = runExternalCommand(
            command = cmakeConfigureCommand(cmake, hostDir, buildDir, options),
            workingDir = projectDir,
            logPath = buildDir.resolve("cmake-configure.log"),
            context = context
        )
        if (configure.exitCode != 0) {
            return failedExternalStep(
                code = "minerva.host_verification.cmake_configure_failed",
                message = "CMake configuration failed for the packaged Minerva host project.",
                tolerance = tolerance,
                expectedOutput = expectedOutput,
                result = configure,
                logPath = buildDir.resolve("cmake-configure.log")
            )
        }
        val build = runExternalCommand(
            command = listOf(cmake, "--build", buildDir.toString()),
            workingDir = projectDir,
            logPath = buildDir.resolve("cmake-build.log"),
            context = context
        )
        if (build.exitCode != 0) {
            return failedExternalStep(
                code = "minerva.host_verification.cmake_build_failed",
                message = "CMake build failed for the packaged Minerva host project.",
                tolerance = tolerance,
                expectedOutput = expectedOutput,
                result = build,
                logPath = buildDir.resolve("cmake-build.log")
            )
        }
        return null
    }

    private fun cmakeConfigureCommand(
        cmake: String,
        hostDir: Path,
        buildDir: Path,
        options: MinervaExportOptions
    ): List<String> {
        val command = mutableListOf(
            cmake,
            "-S",
            hostDir.toString(),
            "-B",
            buildDir.toString(),
            "-DBUILD_TESTING=ON"
        )
        options.runtimeRoot?.let { command += "-DMINERVA_RUNTIME_ROOT=${Paths.get(it).toAbsolutePath().normalize()}" }
        options.metadata[MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE]?.let {
            command += "-DMINERVA_HOST_ADAPTER_SOURCE=${Paths.get(it).toAbsolutePath().normalize()}"
        }
        options.metadata[MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS]?.let {
            command += "-DMINERVA_HOST_INCLUDE_DIRS=$it"
        }
        options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS]?.let {
            command += "-DMINERVA_HOST_LIBRARY_DIRS=$it"
        }
        options.metadata[MinervaHostVerificationMetadata.HOST_LIBRARIES]?.let {
            command += "-DMINERVA_HOST_LIBRARIES=$it"
        }
        return command
    }

    private fun runCTest(
        projectDir: Path,
        options: MinervaExportOptions,
        context: GraphExportContext,
        tolerance: Float,
        expectedOutput: List<Float>
    ): MinervaHostVerification? {
        val buildDir = projectDir.resolve("host/build")
        val ctest = options.metadata[MinervaHostVerificationMetadata.CTEST_EXECUTABLE] ?: "ctest"
        val result = runExternalCommand(
            command = listOf(ctest, "--test-dir", buildDir.toString(), "--output-on-failure"),
            workingDir = projectDir,
            logPath = buildDir.resolve("ctest.log"),
            context = context
        )
        if (result.exitCode != 0) {
            return failedExternalStep(
                code = "minerva.host_verification.ctest_failed",
                message = "CTest failed for the packaged Minerva host project.",
                tolerance = tolerance,
                expectedOutput = expectedOutput,
                hostBuildStatus = MinervaHostVerificationStatus.PASSED,
                hostRunStatus = MinervaHostVerificationStatus.FAILED,
                result = result,
                logPath = buildDir.resolve("ctest.log")
            )
        }
        return null
    }

    private fun failedExternalStep(
        code: String,
        message: String,
        tolerance: Float,
        expectedOutput: List<Float>,
        hostBuildStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.FAILED,
        hostRunStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
        result: ProcessResult,
        logPath: Path
    ): MinervaHostVerification {
        return failed(
            code = code,
            message = message,
            tolerance = tolerance,
            expectedOutput = expectedOutput,
            hostBuildStatus = hostBuildStatus,
            hostRunStatus = hostRunStatus,
            parityStatus = MinervaHostVerificationStatus.SKIPPED,
            remediation = "Inspect the host verification log and ensure CMake, compiler, and libminerva paths are configured.",
            details = mapOf(
                "exitCode" to result.exitCode.toString(),
                "logPath" to logPath.toString(),
                "stdout" to excerpt(result.stdout),
                "stderr" to excerpt(result.stderr)
            )
        )
    }

    private fun runExternalCommand(
        command: List<String>,
        workingDir: Path,
        logPath: Path,
        context: GraphExportContext
    ): ProcessResult {
        return try {
            Files.createDirectories(logPath.parent)
            val result = runProcess(command, workingDir)
            Files.writeString(
                logPath,
                buildString {
                    appendLine("command: ${command.joinToString(" ")}")
                    appendLine("exitCode: ${result.exitCode}")
                    appendLine()
                    appendLine("stdout:")
                    appendLine(result.stdout)
                    appendLine("stderr:")
                    appendLine(result.stderr)
                }
            )
            context.addArtifact(
                GraphExportArtifact(
                    path = logPath.toString(),
                    role = GraphExportArtifactRole.LOG,
                    description = "Minerva host verification log"
                )
            )
            result
        } catch (exception: IOException) {
            ProcessResult(
                exitCode = -1,
                stdout = "",
                stderr = exception.message ?: exception.toString()
            )
        } catch (exception: InterruptedException) {
            Thread.currentThread().interrupt()
            ProcessResult(
                exitCode = -1,
                stdout = "",
                stderr = exception.message ?: exception.toString()
            )
        }
    }

    private fun runProcess(command: List<String>, workingDir: Path): ProcessResult {
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
    }

    private fun readFloatOutput(path: Path): List<Float> {
        if (!Files.isRegularFile(path)) {
            throw IllegalArgumentException("host output file does not exist: $path")
        }
        val tokens = Files.readString(path)
            .trim()
            .split(Regex("[,\\s]+"))
            .filter { it.isNotBlank() }
        require(tokens.isNotEmpty()) { "host output file does not contain numeric values" }
        return tokens.map { token ->
            token.toFloatOrNull()?.takeIf { it.isFinite() }
                ?: throw IllegalArgumentException("host output token is not a finite float: $token")
        }
    }

    private fun resolveProjectPath(projectDir: Path, value: String): Path {
        val path = Paths.get(value)
        return if (path.isAbsolute) path.normalize() else projectDir.resolve(path).normalize()
    }

    private fun metadataFlag(options: MinervaExportOptions, key: String): Boolean {
        return options.metadata[key]?.equals("true", ignoreCase = true) == true
    }

    private fun failed(
        code: String,
        message: String,
        tolerance: Float,
        hostBuildStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
        hostRunStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.SKIPPED,
        parityStatus: MinervaHostVerificationStatus = MinervaHostVerificationStatus.FAILED,
        maxAbsoluteError: Float? = null,
        expectedOutput: List<Float> = emptyList(),
        observedOutput: List<Float> = emptyList(),
        remediation: String,
        details: Map<String, String> = emptyMap()
    ): MinervaHostVerification {
        return MinervaHostVerification(
            status = MinervaHostVerificationStatus.FAILED,
            code = code,
            message = message,
            hostBuildStatus = hostBuildStatus,
            hostRunStatus = hostRunStatus,
            parityStatus = parityStatus,
            tolerance = tolerance,
            maxAbsoluteError = maxAbsoluteError,
            expectedOutput = expectedOutput,
            observedOutput = observedOutput,
            remediation = remediation,
            details = details
        )
    }

    private fun excerpt(value: String, limit: Int = 2000): String {
        return if (value.length <= limit) value else value.take(limit) + "...<truncated>"
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
