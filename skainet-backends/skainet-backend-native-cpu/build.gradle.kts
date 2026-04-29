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

kotlin.sourceSets.named("jvmMain") {
    resources.srcDir(nativeResourcesRoot)
}

tasks.named("jvmProcessResources") {
    dependsOn(packageNativeKernels)
}

// Forward `-Dskainet.runBench=true` from Gradle CLI to the forked test
// JVM so Q4KMatmulMicrobenchTest activates. Skipped silently otherwise.
val runBenchProperty = providers.systemProperty("skainet.runBench")

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
    runBenchProperty.orNull?.let { systemProperty("skainet.runBench", it) }
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
}
