package sk.ainet.compile.minerva

import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportContext

class MinervaJvmCompilerAndPackagerTest {

    @Test
    fun pythonAdapterFailsBeforeProcessWhenCompilerScriptMissing() {
        val (options, intermediate, npzModel) = artifacts(
            outputDir = tempDir("missing-script").toString(),
            projectName = "MissingScript"
        )
        val exception = assertFailsWith<MinervaCompilerException> {
            PythonMinervaCompilerAdapter().compile(
                MinervaCompilerRequest(options, intermediate, npzModel),
                minervaContext(options)
            )
        }

        assertEquals("minerva.compiler.script_missing", exception.code)
        assertTrue(exception.prerequisite)
        assertTrue(exception.remediation.contains("compilerScript"))
    }

    @Test
    fun pythonAdapterInvokesConfiguredExecutableAndReturnsWeights() {
        val root = tempDir("fake-compiler")
        val fakeExecutable = root.resolve("fake-python")
        val compilerScript = root.resolve("compile.py")
        Files.writeString(compilerScript, "# fake script marker\n")
        Files.writeString(
            fakeExecutable,
            """
                |#!/bin/sh
                |out=""
                |while [ "${'$'}#" -gt 0 ]; do
                |  case "${'$'}1" in
                |    --out-dir)
                |      shift
                |      out="${'$'}1"
                |      ;;
                |  esac
                |  shift
                |done
                |mkdir -p "${'$'}out"
                |printf '%s\n' 'int minerva_weights = 1;' > "${'$'}out/weights.c"
                |printf '%s\n' '#pragma once' > "${'$'}out/weights.h"
                |printf '%s\n' 'compiler ok'
                |
            """.trimMargin()
        )
        assertTrue(fakeExecutable.toFile().setExecutable(true))

        val (options, intermediate, npzModel) = artifacts(
            outputDir = root.resolve("out").toString(),
            projectName = "AdapterMlp",
            compilerScript = compilerScript.toString(),
            pythonExecutable = fakeExecutable.toString()
        )
        val context = minervaContext(options)

        val output = PythonMinervaCompilerAdapter().compile(
            MinervaCompilerRequest(options, intermediate, npzModel),
            context
        )

        assertEquals(0, output.exitCode)
        assertTrue(Files.isRegularFile(Path.of(output.weightsCPath)))
        assertTrue(Files.isRegularFile(Path.of(output.weightsHPath)))
        assertTrue(output.stdout.contains("compiler ok"))
        assertTrue(output.commandSummary.contains("model.npz"))
        assertTrue(output.commandSummary.contains("--quant q8"))
        assertFalse(output.commandSummary.contains("--model"))
        assertFalse(output.commandSummary.contains("--quantization"))
        assertTrue(context.diagnostics.any { it.code == "minerva.compiler.completed" })
        assertTrue(context.artifacts.any { it.role == GraphExportArtifactRole.SOURCE && it.path.endsWith("weights.c") })
    }

    @Test
    fun pythonAdapterUsesCurrentLibminervaCompilerCli() {
        val root = tempDir("compiler-cli")
        val fakeExecutable = root.resolve("fake-python")
        val compilerScript = root.resolve("minerva_compile.py")
        val keyFile = root.resolve("device.key")
        val calibrationNpz = root.resolve("calibration.npz")
        Files.writeString(compilerScript, "# fake script marker\n")
        Files.write(keyFile, ByteArray(32) { index -> index.toByte() })
        Files.write(calibrationNpz, byteArrayOf(1, 2, 3))
        Files.writeString(
            fakeExecutable,
            """
                |#!/bin/sh
                |out=""
                |while [ "${'$'}#" -gt 0 ]; do
                |  case "${'$'}1" in
                |    --out-dir)
                |      shift
                |      out="${'$'}1"
                |      ;;
                |  esac
                |  shift
                |done
                |mkdir -p "${'$'}out"
                |printf '%s\n' 'int minerva_weights = 1;' > "${'$'}out/weights.c"
                |printf '%s\n' '#pragma once' > "${'$'}out/weights.h"
                |
            """.trimMargin()
        )
        assertTrue(fakeExecutable.toFile().setExecutable(true))

        val (options, intermediate, npzModel) = artifacts(
            outputDir = root.resolve("out").toString(),
            projectName = "CompilerCliMlp",
            compilerScript = compilerScript.toString(),
            pythonExecutable = fakeExecutable.toString(),
            keyFile = keyFile.toString(),
            calibrationNpz = calibrationNpz.toString()
        )

        val output = PythonMinervaCompilerAdapter().compile(
            MinervaCompilerRequest(options, intermediate, npzModel),
            minervaContext(options)
        )

        assertTrue(output.commandSummary.contains("minerva_compile.py"))
        assertTrue(output.commandSummary.contains("model.npz --out-dir"))
        assertTrue(output.commandSummary.contains("--target atmega328p"))
        assertTrue(output.commandSummary.contains("--quant q8"))
        assertTrue(output.commandSummary.contains("--key <key-file>"))
        assertTrue(output.commandSummary.contains("--calibrate"))
        assertFalse(output.commandSummary.contains(keyFile.toString()))
        assertFalse(output.commandSummary.contains("--key-file"))
        assertFalse(output.commandSummary.contains("--runtime-root"))
        assertEquals(0, output.exitCode)
    }

    @Test
    fun projectPackagerWritesManifestSamplesAndSecretTemplate() {
        val root = tempDir("packager")
        val compilerDir = root.resolve("compiler")
        Files.createDirectories(compilerDir)
        val weightsC = compilerDir.resolve("weights.c")
        val weightsH = compilerDir.resolve("weights.h")
        val debugWeights = compilerDir.resolve("weights_debug.npz")
        Files.writeString(weightsC, "int minerva_weights = 1;\n")
        Files.writeString(weightsH, "#pragma once\n")
        Files.write(debugWeights, byteArrayOf(1, 2, 3, 4))
        val keyFile = root.resolve("device.key")
        Files.writeString(keyFile, "REAL_SECRET_KEY_MATERIAL")
        val (options, intermediate, npzModel) = artifacts(
            outputDir = root.resolve("package").toString(),
            projectName = "PackagedJvmMlp",
            compilerScript = root.resolve("compile.py").toString(),
            keyFile = keyFile.toString()
        )
        val compilerOutput = MinervaCompilerOutput(
            outputDir = compilerDir.toString(),
            weightsCPath = weightsC.toString(),
            weightsHPath = weightsH.toString(),
            debugWeightsPath = debugWeights.toString(),
            commandSummary = "fake-minerva model.npz --key <key-file>"
        )
        val context = minervaContext(options)

        val bundle = JvmMinervaProjectPackager().packageProject(
            MinervaProjectPackageRequest(options, intermediate, npzModel, compilerOutput),
            context
        )

        val projectDir = Path.of(bundle.outputDir)
        assertTrue(Files.isRegularFile(projectDir.resolve("manifest.json")))
        assertTrue(Files.isRegularFile(projectDir.resolve("generated/model.npz")))
        assertTrue(Files.isRegularFile(projectDir.resolve("generated/weights.c")))
        assertTrue(Files.isRegularFile(projectDir.resolve("generated/weights_debug.npz")))
        assertTrue(Files.isRegularFile(projectDir.resolve("include/weights.h")))
        assertTrue(Files.isRegularFile(projectDir.resolve("host/reference-input.txt")))
        assertTrue(Files.isRegularFile(projectDir.resolve("host/reference-output.txt")))
        assertTrue(Files.isRegularFile(projectDir.resolve("host/main.c")))
        assertTrue(Files.isRegularFile(projectDir.resolve("host/runtime_adapter.example.c")))
        assertTrue(Files.isRegularFile(projectDir.resolve("firmware/main.c")))
        val hostCmake = Files.readString(projectDir.resolve("host/CMakeLists.txt"))
        assertTrue(hostCmake.contains("MINERVA_HOST_ADAPTER_SOURCE"))
        assertTrue(hostCmake.contains("add_subdirectory"))
        assertTrue(hostCmake.contains("minerva_define_from_weights"))
        assertTrue(hostCmake.contains("target_sources(minerva"))
        assertTrue(hostCmake.contains("target_link_libraries"))
        assertTrue(hostCmake.contains("set_tests_properties"))
        val hostMain = Files.readString(projectDir.resolve("host/main.c"))
        assertTrue(hostMain.contains("0.25f, 0.5f, 0.75f, 1.0f"))
        assertTrue(hostMain.contains("minerva_run_inference"))
        assertTrue(hostMain.contains("observed-output.txt"))
        val adapterExample = Files.readString(projectDir.resolve("host/runtime_adapter.example.c"))
        assertTrue(adapterExample.contains("minerva_run_inference"))
        assertTrue(adapterExample.contains("mnv_init"))
        assertTrue(adapterExample.contains("mnv_seed_prng"))
        assertTrue(adapterExample.contains("mnv_run_with_model"))
        assertTrue(adapterExample.contains("mnv_run"))
        assertTrue(adapterExample.contains("mnv_verify_output_with_key"))
        assertTrue(adapterExample.contains("mnv_verify_output"))
        assertTrue(adapterExample.contains("MNV_INPUT_SIZE"))
        assertTrue(adapterExample.contains("MNV_OUTPUT_SIZE"))
        assertFalse(adapterExample.contains("return 1;"))
        val secretsExample = Files.readString(projectDir.resolve("include/secrets.example.h"))
        assertTrue(secretsExample.contains("replace-with-device-key"))
        assertFalse(secretsExample.contains("REAL_SECRET_KEY_MATERIAL"))
        val manifest = Files.readString(projectDir.resolve("manifest.json"))
        assertTrue(manifest.contains("\"target\": \"atmega328p\""))
        assertTrue(manifest.contains("\"compilerCommand\": \"fake-minerva model.npz --key <key-file>\""))
        assertTrue(manifest.contains("\"referenceInputPath\": \"host/reference-input.txt\""))
        assertTrue(manifest.contains("\"referenceOutputPath\": \"host/reference-output.txt\""))
        assertTrue(manifest.contains("\"generatedFileSha256\": {"))
        assertTrue(Regex("\"generated/model\\.npz\": \"[0-9a-f]{64}\"").containsMatchIn(manifest))
        assertTrue(Regex("\"generated/weights\\.c\": \"[0-9a-f]{64}\"").containsMatchIn(manifest))
        assertTrue(Regex("\"include/weights\\.h\": \"[0-9a-f]{64}\"").containsMatchIn(manifest))
        assertFalse(manifest.contains("REAL_SECRET_KEY_MATERIAL"))
        assertEquals("manifest.json", bundle.manifestPath)
        assertTrue(bundle.generatedFiles.contains("generated/weights_debug.npz"))
        assertTrue(bundle.generatedFiles.contains("include/secrets.example.h"))
        assertTrue(bundle.generatedFiles.contains("host/reference-input.txt"))
        assertTrue(bundle.generatedFiles.contains("host/reference-output.txt"))
        assertTrue(bundle.generatedFiles.contains("host/runtime_adapter.example.c"))
        assertTrue(context.artifacts.any { it.role == GraphExportArtifactRole.PROJECT_DIRECTORY })
        assertTrue(context.artifacts.any { it.role == GraphExportArtifactRole.TEST_REPORT })
        assertTrue(context.diagnostics.any { it.code == "minerva.packaging.completed" })
    }

    @Test
    fun projectPackagerNormalizesCurrentLibminervaLayerClosers() {
        val root = tempDir("packager-libminerva-normalize")
        val outputRoot = root.resolve("package")
        val projectName = "NormalizePackagedMlp"
        val generatedDir = outputRoot.resolve(projectName).resolve("generated")
        val compilerDir = root.resolve("compiler")
        Files.createDirectories(generatedDir)
        Files.createDirectories(compilerDir)
        val weightsC = generatedDir.resolve("weights.c")
        val weightsH = compilerDir.resolve("weights.h")
        Files.writeString(
            weightsC,
            """
                |const mnv_layer_desc_t mnv_layers[2] PROGMEM = {
                |    [0] = {
                |        .weights = NULL,
                |    }},
                |    [1] = {
                |        .weights = NULL,
                |    }},
                |};
                |
            """.trimMargin()
        )
        Files.writeString(weightsH, "#pragma once\n")
        val (options, intermediate, npzModel) = artifacts(
            outputDir = outputRoot.toString(),
            projectName = projectName,
            compilerScript = root.resolve("compile.py").toString()
        )
        val compilerOutput = MinervaCompilerOutput(
            outputDir = compilerDir.toString(),
            weightsCPath = weightsC.toString(),
            weightsHPath = weightsH.toString(),
            commandSummary = "fake-minerva model.npz"
        )

        val bundle = JvmMinervaProjectPackager().packageProject(
            MinervaProjectPackageRequest(options, intermediate, npzModel, compilerOutput),
            minervaContext(options)
        )

        val packagedWeights = Files.readString(Path.of(bundle.outputDir).resolve("generated/weights.c"))
        assertFalse(packagedWeights.contains("}},"))
        assertTrue(packagedWeights.contains("    },\n    [1] = {"))
    }

    @Test
    fun hostRuntimeAdapterExampleCompilesAgainstPublicMinervaApiShim() {
        val root = tempDir("adapter-compile")
        val fixture = packagedProject(root, "AdapterCompile")
        val includeDir = fixture.projectDir.resolve("include")
        val runtimeIncludeDir = root.resolve("runtime/include")
        Files.createDirectories(runtimeIncludeDir)
        Files.writeString(
            includeDir.resolve("weights.h"),
            """
                |#pragma once
                |#include "minerva.h"
                |#define MNV_INPUT_SIZE 4U
                |#define MNV_OUTPUT_SIZE 3U
                |extern const mnv_model_t mnv_model;
                |
            """.trimMargin()
        )
        Files.writeString(
            runtimeIncludeDir.resolve("minerva.h"),
            """
                |#pragma once
                |#include <stdint.h>
                |#define MNV_OK 0
                |#define MNV_ENABLE_OUTPUT_AUTH 1
                |typedef int mnv_status_t;
                |typedef int8_t mnv_act_t;
                |typedef struct mnv_ctx_t { int verified; } mnv_ctx_t;
                |typedef struct mnv_model_t { int version; const uint8_t *key; } mnv_model_t;
                |mnv_status_t mnv_init(mnv_ctx_t *ctx, const mnv_model_t *model);
                |void mnv_seed_prng(mnv_ctx_t *ctx, uint32_t seed);
                |mnv_status_t mnv_run(mnv_ctx_t *ctx, const mnv_act_t *input, mnv_act_t *output);
                |mnv_status_t mnv_run_with_model(
                |    mnv_ctx_t *ctx,
                |    const mnv_model_t *model,
                |    const mnv_act_t *input,
                |    mnv_act_t *output
                |);
                |mnv_status_t mnv_verify_output(const mnv_ctx_t *ctx, const mnv_act_t *input, const mnv_act_t *output);
                |mnv_status_t mnv_verify_output_with_key(
                |    const mnv_ctx_t *ctx,
                |    const uint8_t *device_key,
                |    const mnv_act_t *input,
                |    const mnv_act_t *output
                |);
                |
            """.trimMargin()
        )

        val process = ProcessBuilder(
            "cc",
            "-std=c11",
            "-Wall",
            "-Werror",
            "-I${runtimeIncludeDir}",
            "-I${includeDir}",
            "-c",
            fixture.projectDir.resolve("host/runtime_adapter.example.c").toString(),
            "-o",
            root.resolve("runtime_adapter.example.o").toString()
        ).start()
        val stdout = process.inputStream.bufferedReader().readText()
        val stderr = process.errorStream.bufferedReader().readText()

        assertEquals(0, process.waitFor(), stdout + stderr)
        assertTrue(Files.isRegularFile(root.resolve("runtime_adapter.example.o")))
    }

    @Test
    fun hostVerifierPassesStructuralAndReferenceChecksWithoutExternalRuntime() {
        val fixture = packagedProject(tempDir("host-verify-lightweight"), "LightweightVerify")

        val verification = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        assertEquals(MinervaHostVerificationStatus.SKIPPED, verification.hostBuildStatus)
        assertEquals(MinervaHostVerificationStatus.SKIPPED, verification.hostRunStatus)
        assertEquals(MinervaHostVerificationStatus.SKIPPED, verification.parityStatus)
        assertTrue(verification.expectedOutput.isNotEmpty())
        assertTrue(fixture.context.diagnostics.any { it.code == "minerva.host_verification.passed" })
    }

    @Test
    fun hostVerifierHandlesBinaryKeyMaterialDuringSecretHygieneCheck() {
        val root = tempDir("host-verify-binary-key")
        val keyFile = root.resolve("device.key")
        Files.write(keyFile, ByteArray(32) { index -> (index * 7).toByte() })
        val fixture = packagedProject(root, "BinaryKeyVerify")
        val request = fixture.request.copy(
            options = fixture.request.options.copy(keyFile = keyFile.toString())
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
    }

    @Test
    fun hostVerifierPassesAdapterConfigurationToCMake() {
        val fixture = packagedProject(tempDir("host-verify-cmake-adapter"), "AdapterCmakeVerify")
        val toolDir = tempDir("fake-cmake")
        val fakeCmake = toolDir.resolve("cmake")
        Files.writeString(
            fakeCmake,
            """
                |#!/bin/sh
                |exit 0
                |
            """.trimMargin()
        )
        assertTrue(fakeCmake.toFile().setExecutable(true))
        val runtimeRoot = toolDir.resolve("runtime")
        val adapterSource = toolDir.resolve("minerva_adapter.c")
        Files.createDirectories(runtimeRoot)
        Files.writeString(adapterSource, "int minerva_run_inference(void) { return 0; }\n")
        val request = fixture.request.copy(
            options = fixture.request.options.copy(
                runtimeRoot = runtimeRoot.toString(),
                metadata = fixture.request.options.metadata + mapOf(
                    MinervaHostVerificationMetadata.RUN_CMAKE_BUILD to "true",
                    MinervaHostVerificationMetadata.CMAKE_EXECUTABLE to fakeCmake.toString(),
                    MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE to adapterSource.toString(),
                    MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS to "/opt/libminerva/include;/project/include",
                    MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS to "/opt/libminerva/lib",
                    MinervaHostVerificationMetadata.HOST_LIBRARIES to "minerva;crypto"
                )
            )
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.hostBuildStatus)
        val log = Files.readString(fixture.projectDir.resolve("host/build/cmake-configure.log"))
        assertTrue(log.contains("-DMINERVA_RUNTIME_ROOT=${runtimeRoot.toAbsolutePath().normalize()}"))
        assertTrue(log.contains("-DMINERVA_HOST_ADAPTER_SOURCE=${adapterSource.toAbsolutePath().normalize()}"))
        assertTrue(log.contains("-DMINERVA_HOST_INCLUDE_DIRS=/opt/libminerva/include;/project/include"))
        assertTrue(log.contains("-DMINERVA_HOST_LIBRARY_DIRS=/opt/libminerva/lib"))
        assertTrue(log.contains("-DMINERVA_HOST_LIBRARIES=minerva;crypto"))
    }

    @Test
    fun hostVerifierUsesAbsoluteCmakePathsForRelativeBundleOutput() {
        val relativeRoot = Path.of("build/minerva-relative-${System.nanoTime()}")
        val fixture = packagedProject(relativeRoot, "RelativeCmakeVerify")
        val toolDir = tempDir("fake-cmake-relative")
        val fakeCmake = toolDir.resolve("cmake")
        Files.writeString(
            fakeCmake,
            """
                |#!/bin/sh
                |exit 0
                |
            """.trimMargin()
        )
        assertTrue(fakeCmake.toFile().setExecutable(true))
        val request = fixture.request.copy(
            options = fixture.request.options.copy(
                metadata = fixture.request.options.metadata + mapOf(
                    MinervaHostVerificationMetadata.RUN_CMAKE_BUILD to "true",
                    MinervaHostVerificationMetadata.CMAKE_EXECUTABLE to fakeCmake.toString()
                )
            )
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        val log = Files.readString(fixture.projectDir.resolve("host/build/cmake-configure.log"))
        assertTrue(log.contains("-S ${fixture.projectDir.resolve("host").toAbsolutePath().normalize()}"))
        assertTrue(log.contains("-B ${fixture.projectDir.resolve("host/build").toAbsolutePath().normalize()}"))
    }

    @Test
    fun hostVerifierBuildsRuntimeCheckoutAndRunsCTest() {
        if (!commandAvailable("cmake")) return
        val root = tempDir("host-verify-runtime-checkout")
        val fixture = packagedProject(root, "CheckoutRuntimeVerify")
        val secretIncludeDir = root.resolve("secret-include")
        val runtimeRoot = root.resolve("libminerva")
        Files.createDirectories(secretIncludeDir)
        Files.writeString(secretIncludeDir.resolve("secrets.h"), "#pragma once\n")
        writeFakeRuntimeCheckout(runtimeRoot)
        Files.writeString(
            fixture.projectDir.resolve("include/weights.h"),
            """
                |#pragma once
                |#include "minerva.h"
                |#define MNV_INPUT_SIZE 4U
                |#define MNV_NUM_LAYERS 2U
                |#define MNV_LAYER_0_SIZE 3U
                |#define MNV_LAYER_1_SIZE 2U
                |#define MNV_OUTPUT_SIZE 3U
                |extern const mnv_model_t mnv_model;
                |
            """.trimMargin()
        )
        Files.writeString(
            fixture.projectDir.resolve("generated/weights.c"),
            """
                |#include "weights.h"
                |#include "secrets.h"
                |const mnv_model_t mnv_model = {0};
                |
            """.trimMargin()
        )
        val request = fixture.request.copy(
            options = fixture.request.options.copy(
                runtimeRoot = runtimeRoot.toString(),
                hostVerificationTolerance = 1.0f,
                metadata = fixture.request.options.metadata + mapOf(
                    MinervaHostVerificationMetadata.RUN_CMAKE_BUILD to "true",
                    MinervaHostVerificationMetadata.RUN_CTEST to "true",
                    MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE to
                        fixture.projectDir.resolve("host/runtime_adapter.example.c").toString(),
                    MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS to secretIncludeDir.toString()
                )
            )
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.hostBuildStatus)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.hostRunStatus)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.parityStatus)
        assertEquals(3, verification.observedOutput.size)
        assertTrue(Files.readString(fixture.projectDir.resolve("host/build/ctest.log")).contains("100% tests passed"))
    }

    @Test
    fun hostVerifierComparesConfiguredHostOutput() {
        val fixture = packagedProject(tempDir("host-verify-output"), "OutputVerify")
        val baseline = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)
        val hostOutputPath = fixture.projectDir.resolve("host-output.txt")
        Files.writeString(hostOutputPath, baseline.expectedOutput.joinToString(separator = "\n"))
        val request = fixture.request.copy(
            options = fixture.request.options.copy(
                metadata = fixture.request.options.metadata +
                    (MinervaHostVerificationMetadata.HOST_OUTPUT_PATH to "host-output.txt")
            )
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.hostRunStatus)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.parityStatus)
        assertEquals(0.0f, verification.maxAbsoluteError)
        assertEquals(baseline.expectedOutput, verification.observedOutput)
    }

    @Test
    fun hostVerifierUsesDefaultObservedOutputPath() {
        val fixture = packagedProject(tempDir("host-verify-default-output"), "DefaultOutputVerify")
        val baseline = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)
        Files.writeString(
            fixture.projectDir.resolve("host/observed-output.txt"),
            baseline.expectedOutput.joinToString(separator = "\n")
        )

        val verification = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.PASSED, verification.status)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.hostRunStatus)
        assertEquals(MinervaHostVerificationStatus.PASSED, verification.parityStatus)
        assertEquals(baseline.expectedOutput, verification.observedOutput)
    }

    @Test
    fun hostVerifierFailsWhenHostOutputExceedsTolerance() {
        val fixture = packagedProject(tempDir("host-verify-mismatch"), "MismatchVerify")
        Files.writeString(fixture.projectDir.resolve("host-output.txt"), "999 999 999")
        val request = fixture.request.copy(
            options = fixture.request.options.copy(
                metadata = fixture.request.options.metadata +
                    (MinervaHostVerificationMetadata.HOST_OUTPUT_PATH to "host-output.txt")
            )
        )

        val verification = JvmMinervaHostVerifier().verify(request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.FAILED, verification.status)
        assertEquals("minerva.host_verification.parity_failed", verification.code)
        assertEquals(MinervaHostVerificationStatus.FAILED, verification.parityStatus)
        assertTrue((verification.maxAbsoluteError ?: 0.0f) > verification.tolerance)
    }

    @Test
    fun hostVerifierFailsWhenReferenceOutputFixtureDrifts() {
        val fixture = packagedProject(tempDir("host-verify-reference-drift"), "ReferenceDriftVerify")
        Files.writeString(fixture.projectDir.resolve("host/reference-output.txt"), "999 999 999")

        val verification = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.FAILED, verification.status)
        assertEquals("minerva.host_verification.reference_output_mismatch", verification.code)
        assertEquals(MinervaHostVerificationStatus.FAILED, verification.parityStatus)
        assertTrue((verification.maxAbsoluteError ?: 0.0f) > verification.tolerance)
    }

    @Test
    fun hostVerifierFailsWhenPackagedModelWasTampered() {
        val fixture = packagedProject(tempDir("host-verify-tampered"), "TamperedVerify")
        Files.write(fixture.projectDir.resolve("generated/model.npz"), byteArrayOf(0, 1, 2, 3))

        val verification = JvmMinervaHostVerifier().verify(fixture.request, fixture.context)

        assertEquals(MinervaHostVerificationStatus.FAILED, verification.status)
        assertEquals("minerva.host_verification.model_tampered", verification.code)
    }

    private fun artifacts(
        outputDir: String,
        projectName: String,
        compilerScript: String? = null,
        pythonExecutable: String = "python3",
        keyFile: String? = null,
        calibrationNpz: String? = null
    ): Triple<MinervaExportOptions, MinervaIntermediate, MinervaNpzModel> {
        val options = minervaTestOptions(
            outputDir = outputDir,
            projectName = projectName
        ).copy(
            compilerScript = compilerScript,
            pythonExecutable = pythonExecutable,
            keyFile = keyFile,
            calibrationNpz = calibrationNpz,
            runHostVerification = false
        )
        val context = minervaContext(options)
        val intermediate = MinervaGraphCanonicalizer().convert(validMinervaMlpGraph(), context)
        val npzModel = MinervaNpzModelWriter().write(intermediate, context)
        return Triple(options, intermediate, npzModel)
    }

    private fun minervaContext(options: MinervaExportOptions): GraphExportContext {
        return GraphExportContext(
            backendName = MinervaExportBackend.backendName,
            targetName = options.projectName,
            metadata = options.toMetadata()
        )
    }

    private fun packagedProject(root: Path, projectName: String): PackagedProjectFixture {
        val compilerDir = root.resolve("compiler")
        Files.createDirectories(compilerDir)
        val weightsC = compilerDir.resolve("weights.c")
        val weightsH = compilerDir.resolve("weights.h")
        Files.writeString(weightsC, "int minerva_weights = 1;\n")
        Files.writeString(weightsH, "#pragma once\nextern int minerva_weights;\n")
        val (options, intermediate, npzModel) = artifacts(
            outputDir = root.resolve("package").toString(),
            projectName = projectName,
            compilerScript = root.resolve("compile.py").toString()
        )
        val compilerOutput = MinervaCompilerOutput(
            outputDir = compilerDir.toString(),
            weightsCPath = weightsC.toString(),
            weightsHPath = weightsH.toString(),
            commandSummary = "fake-minerva model.npz"
        )
        val context = minervaContext(options)
        val bundle = JvmMinervaProjectPackager().packageProject(
            MinervaProjectPackageRequest(options, intermediate, npzModel, compilerOutput),
            context
        )
        val request = MinervaHostVerificationRequest(
            options = options,
            intermediate = intermediate,
            npzModel = npzModel,
            compilerOutput = compilerOutput,
            bundle = bundle
        )
        return PackagedProjectFixture(
            request = request,
            context = context,
            projectDir = Path.of(bundle.outputDir)
        )
    }

    private fun tempDir(prefix: String): Path {
        return Files.createTempDirectory("skainet-minerva-$prefix-")
    }

    private fun commandAvailable(command: String): Boolean {
        return try {
            ProcessBuilder(command, "--version").start().waitFor() == 0
        } catch (_: IOException) {
            false
        }
    }

    private fun writeFakeRuntimeCheckout(root: Path) {
        val includeDir = root.resolve("include")
        val sourceDir = root.resolve("src")
        Files.createDirectories(includeDir)
        Files.createDirectories(sourceDir)
        Files.writeString(
            root.resolve("CMakeLists.txt"),
            """
                |cmake_minimum_required(VERSION 3.20)
                |project(FakeMinerva C)
                |add_library(minerva STATIC src/minerva.c)
                |target_include_directories(minerva PUBLIC include)
                |
            """.trimMargin()
        )
        Files.writeString(
            includeDir.resolve("minerva.h"),
            """
                |#pragma once
                |#include <stdint.h>
                |#define MNV_OK 0
                |#define MNV_ENABLE_OUTPUT_AUTH 1
                |#ifndef MNV_INPUT_SIZE
                |#error "MNV_INPUT_SIZE was not propagated"
                |#endif
                |#if MNV_INPUT_SIZE != 4U
                |#error "MNV_INPUT_SIZE did not come from generated weights.h"
                |#endif
                |#if MNV_OUTPUT_SIZE != 3U
                |#error "MNV_OUTPUT_SIZE did not come from generated weights.h"
                |#endif
                |typedef int mnv_status_t;
                |typedef int8_t mnv_act_t;
                |typedef struct mnv_ctx_t { int initialized; } mnv_ctx_t;
                |typedef struct mnv_model_t { int version; const uint8_t *key; } mnv_model_t;
                |mnv_status_t mnv_init(mnv_ctx_t *ctx, const mnv_model_t *model);
                |void mnv_seed_prng(mnv_ctx_t *ctx, uint32_t seed);
                |mnv_status_t mnv_run(mnv_ctx_t *ctx, const mnv_act_t *input, mnv_act_t *output);
                |mnv_status_t mnv_run_with_model(
                |    mnv_ctx_t *ctx,
                |    const mnv_model_t *model,
                |    const mnv_act_t *input,
                |    mnv_act_t *output
                |);
                |mnv_status_t mnv_verify_output(const mnv_ctx_t *ctx, const mnv_act_t *input, const mnv_act_t *output);
                |mnv_status_t mnv_verify_output_with_key(
                |    const mnv_ctx_t *ctx,
                |    const uint8_t *device_key,
                |    const mnv_act_t *input,
                |    const mnv_act_t *output
                |);
                |
            """.trimMargin()
        )
        Files.writeString(
            sourceDir.resolve("minerva.c"),
            """
                |#include "minerva.h"
                |
                |mnv_status_t mnv_init(mnv_ctx_t *ctx, const mnv_model_t *model) {
                |    (void)model;
                |    ctx->initialized = 1;
                |    return MNV_OK;
                |}
                |
                |void mnv_seed_prng(mnv_ctx_t *ctx, uint32_t seed) {
                |    (void)ctx;
                |    (void)seed;
                |}
                |
                |mnv_status_t mnv_run(mnv_ctx_t *ctx, const mnv_act_t *input, mnv_act_t *output) {
                |    (void)ctx;
                |    (void)input;
                |    output[0] = 127;
                |    output[1] = 127;
                |    output[2] = 127;
                |    return MNV_OK;
                |}
                |
                |mnv_status_t mnv_run_with_model(
                |    mnv_ctx_t *ctx,
                |    const mnv_model_t *model,
                |    const mnv_act_t *input,
                |    mnv_act_t *output
                |) {
                |    (void)model;
                |    return mnv_run(ctx, input, output);
                |}
                |
                |mnv_status_t mnv_verify_output(const mnv_ctx_t *ctx, const mnv_act_t *input, const mnv_act_t *output) {
                |    (void)ctx;
                |    (void)input;
                |    (void)output;
                |    return MNV_OK;
                |}
                |
                |mnv_status_t mnv_verify_output_with_key(
                |    const mnv_ctx_t *ctx,
                |    const uint8_t *device_key,
                |    const mnv_act_t *input,
                |    const mnv_act_t *output
                |) {
                |    (void)device_key;
                |    return mnv_verify_output(ctx, input, output);
                |}
                |
            """.trimMargin()
        )
    }

    private data class PackagedProjectFixture(
        val request: MinervaHostVerificationRequest,
        val context: GraphExportContext,
        val projectDir: Path
    )
}
