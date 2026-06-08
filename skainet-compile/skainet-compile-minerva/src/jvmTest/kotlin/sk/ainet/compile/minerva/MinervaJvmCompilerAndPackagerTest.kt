package sk.ainet.compile.minerva

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
        assertTrue(output.commandSummary.contains("--model"))
        assertTrue(context.diagnostics.any { it.code == "minerva.compiler.completed" })
        assertTrue(context.artifacts.any { it.role == GraphExportArtifactRole.SOURCE && it.path.endsWith("weights.c") })
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
            commandSummary = "fake-minerva --key-file <key-file>"
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
        assertTrue(hostCmake.contains("target_link_libraries"))
        assertTrue(hostCmake.contains("set_tests_properties"))
        val hostMain = Files.readString(projectDir.resolve("host/main.c"))
        assertTrue(hostMain.contains("0.25f, 0.5f, 0.75f, 1.0f"))
        assertTrue(hostMain.contains("minerva_run_inference"))
        assertTrue(hostMain.contains("observed-output.txt"))
        val adapterExample = Files.readString(projectDir.resolve("host/runtime_adapter.example.c"))
        assertTrue(adapterExample.contains("minerva_run_inference"))
        assertTrue(adapterExample.contains("return 1"))
        val secretsExample = Files.readString(projectDir.resolve("include/secrets.example.h"))
        assertTrue(secretsExample.contains("replace-with-device-key"))
        assertFalse(secretsExample.contains("REAL_SECRET_KEY_MATERIAL"))
        val manifest = Files.readString(projectDir.resolve("manifest.json"))
        assertTrue(manifest.contains("\"target\": \"atmega328p\""))
        assertTrue(manifest.contains("\"compilerCommand\": \"fake-minerva --key-file <key-file>\""))
        assertTrue(manifest.contains("\"referenceInputPath\": \"host/reference-input.txt\""))
        assertTrue(manifest.contains("\"referenceOutputPath\": \"host/reference-output.txt\""))
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
        keyFile: String? = null
    ): Triple<MinervaExportOptions, MinervaIntermediate, MinervaNpzModel> {
        val options = minervaTestOptions(
            outputDir = outputDir,
            projectName = projectName
        ).copy(
            compilerScript = compilerScript,
            pythonExecutable = pythonExecutable,
            keyFile = keyFile,
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
            commandSummary = "fake-minerva --model model.npz"
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

    private data class PackagedProjectFixture(
        val request: MinervaHostVerificationRequest,
        val context: GraphExportContext,
        val projectDir: Path
    )
}
