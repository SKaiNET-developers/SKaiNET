plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

kotlin {
    explicitApi()
    jvm()

    sourceSets {
        val jvmMain by getting {
            dependencies {
                implementation(project(":skainet-backends:skainet-backend-api"))
            }
        }
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                // Parity tests compare NativeQ4KMatmulKernel output
                // against PanamaVectorQ4KMatmulKernel; the Panama
                // kernel pulls in parallelChunks which transitively
                // requires kotlinx-coroutines.
                implementation(project(":skainet-backends:skainet-backend-cpu"))
                implementation(libs.kotlinx.coroutines)
            }
        }
    }
}

// --- Native (CMake) wiring -------------------------------------------------
//
// PR 1 builds for the host arch only. Cross-arch CI matrix is deferred per
// the native-ffm-plan asciidoc. The artifact lands at
//   build/native/resources/native/<os>-<arch>/libskainet_kernels.{so|dylib|dll}
// and is bundled into the JAR via an extra resources srcDir on jvmMain.

val nativeOsArch: String = run {
    val os = System.getProperty("os.name").lowercase()
    val osTag = when {
        os.contains("linux") -> "linux"
        os.contains("mac") || os.contains("darwin") -> "macos"
        os.contains("windows") -> "windows"
        else -> error("Unsupported OS for skainet-backend-native-cpu: $os")
    }
    val archRaw = System.getProperty("os.arch").lowercase()
    val archTag = when (archRaw) {
        "x86_64", "amd64" -> "x86_64"
        "aarch64", "arm64" -> "arm64"
        else -> error("Unsupported arch for skainet-backend-native-cpu: $archRaw")
    }
    "$osTag-$archTag"
}

// Capture as plain Strings up front so configuration-cache serialization
// doesn't have to walk back to the build script for Provider resolution.
val nativeSourcePath: String = layout.projectDirectory.dir("native").asFile.absolutePath
val cmakeBuildPath: String = layout.buildDirectory.dir("native/cmake-build").get().asFile.absolutePath
val nativeResourcesRoot = layout.buildDirectory.dir("native/resources")
val nativeResourceTargetDir = nativeResourcesRoot.map { it.dir("native/$nativeOsArch") }

val configureNativeKernels by tasks.registering(Exec::class) {
    group = "build"
    description = "Run CMake configure step for the native kernels library."
    inputs.file("$nativeSourcePath/CMakeLists.txt")
    inputs.dir("$nativeSourcePath/src")
    inputs.dir("$nativeSourcePath/include")
    outputs.dir(cmakeBuildPath)
    // CMake auto-creates the -B directory; no doFirst mkdirs needed.
    commandLine = listOf(
        "cmake",
        "-S", nativeSourcePath,
        "-B", cmakeBuildPath,
        "-DCMAKE_BUILD_TYPE=Release",
    )
}

val buildNativeKernels by tasks.registering(Exec::class) {
    group = "build"
    description = "Build the native kernels shared library via CMake."
    dependsOn(configureNativeKernels)
    inputs.file("$nativeSourcePath/CMakeLists.txt")
    inputs.dir("$nativeSourcePath/src")
    inputs.dir("$nativeSourcePath/include")
    outputs.dir(cmakeBuildPath)
    commandLine = listOf(
        "cmake",
        "--build", cmakeBuildPath,
        "--config", "Release",
    )
}

val packageNativeKernels by tasks.registering(Copy::class) {
    group = "build"
    description = "Stage the built native kernels library into JVM resources."
    dependsOn(buildNativeKernels)
    from(cmakeBuildPath) {
        include(
            "libskainet_kernels.so",
            "libskainet_kernels.dylib",
            "skainet_kernels.dll",
            "Release/skainet_kernels.dll",
        )
        eachFile { path = name }
    }
    into(nativeResourceTargetDir)
}

// --- Cross-compile to aarch64 (opt-in) -------------------------------------
//
// Produces native/linux-arm64/libskainet_kernels.so with the NEON paths
// (CMAKE_SYSTEM_PROCESSOR=aarch64 -> -march=armv8.2-a+fp16+dotprod). Gated on
// `-PcrossArm64=true` so the default host build is unaffected on machines
// without the `gcc-aarch64-linux-gnu` cross toolchain. The board build / CI
// host opts in. NativeLibraryLoader resolves native/linux-arm64/ from os.arch
// at runtime, so the consuming side needs no change once this .so is bundled.
//
// BOARD-VERIFY-PENDING: the NEON code is syntax-validated for aarch64 but has
// not been executed; run the parity tests under QEMU or on the SL2610.
val crossArm64Enabled: Boolean = (findProperty("crossArm64") as String?)?.toBoolean() == true
val aarch64Cc: String = (findProperty("skainetAarch64Cc") as String?) ?: "aarch64-linux-gnu-gcc"
val cmakeBuildArm64Path: String = layout.buildDirectory.dir("native/cmake-build-arm64").get().asFile.absolutePath
val nativeResourceArm64Dir = nativeResourcesRoot.map { it.dir("native/linux-arm64") }
val toolchainFilePath = "$nativeSourcePath/toolchain-aarch64.cmake"

val configureNativeKernelsArm64 by tasks.registering(Exec::class) {
    group = "build"
    description = "CMake configure for the aarch64 (NEON) native kernels (cross-compile)."
    onlyIf { crossArm64Enabled }
    inputs.file("$nativeSourcePath/CMakeLists.txt")
    inputs.dir("$nativeSourcePath/src")
    inputs.dir("$nativeSourcePath/include")
    outputs.dir(cmakeBuildArm64Path)
    commandLine = listOf(
        "cmake",
        "-S", nativeSourcePath,
        "-B", cmakeBuildArm64Path,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFilePath",
        "-DSKAINET_AARCH64_CC=$aarch64Cc",
    )
}

val buildNativeKernelsArm64 by tasks.registering(Exec::class) {
    group = "build"
    description = "Cross-build the aarch64 (NEON) native kernels shared library."
    onlyIf { crossArm64Enabled }
    dependsOn(configureNativeKernelsArm64)
    inputs.file("$nativeSourcePath/CMakeLists.txt")
    inputs.dir("$nativeSourcePath/src")
    inputs.dir("$nativeSourcePath/include")
    outputs.dir(cmakeBuildArm64Path)
    commandLine = listOf("cmake", "--build", cmakeBuildArm64Path, "--config", "Release")
}

val packageNativeKernelsArm64 by tasks.registering(Copy::class) {
    group = "build"
    description = "Stage the cross-built aarch64 native kernels into JVM resources."
    onlyIf { crossArm64Enabled }
    dependsOn(buildNativeKernelsArm64)
    from(cmakeBuildArm64Path) {
        include("libskainet_kernels.so")
        eachFile { path = name }
    }
    into(nativeResourceArm64Dir)
}

kotlin.sourceSets.named("jvmMain") {
    resources.srcDir(nativeResourcesRoot)
}

tasks.named("jvmProcessResources") {
    dependsOn(packageNativeKernels)
    if (crossArm64Enabled) dependsOn(packageNativeKernelsArm64)
}

// Forward `-Dskainet.runBench=true` from Gradle CLI to the forked test
// JVM so Q4KMatmulMicrobenchTest activates. Skipped silently otherwise.
val runBenchProperty = providers.systemProperty("skainet.runBench")

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
    runBenchProperty.orNull?.let { systemProperty("skainet.runBench", it) }
    // Stable version stamp for KernelSupportMatrixTest's kernel-support.json (avoids doc churn).
    systemProperty("skainet.version", (findProperty("VERSION_NAME") ?: "dev").toString())
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
}
