plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Paths shared by the K/N cinterop (kotlin block) and the CMake tasks below.
val nativeIncludeDir: String = layout.projectDirectory.dir("native/include").asFile.absolutePath
val staticArchivePath: String =
    layout.buildDirectory.file("native/cmake-build/libskainet_kernels.a").get().asFile.absolutePath
// aarch64 cross-built static archive (produced by buildNativeKernelsArm64 with
// -PcrossArm64; carries the NEON paths). Linked into linuxArm64 binaries.
val staticArchiveArm64Path: String =
    layout.buildDirectory.file("native/cmake-build-arm64/libskainet_kernels.a").get().asFile.absolutePath

// --- Archive embedding (#941) ----------------------------------------------
//
// The static archive must be EMBEDDED into the cinterop klib (cinterop
// -staticLibrary/-libraryPath) rather than attached via linkerOpts:
// linkerOpts applies only to this project's own binaries, so a published
// klib would carry bindings but no machine code and no downstream K/N
// consumer could link. Embedding is conditional on a correct-architecture
// ELF archive being available:
//   - linuxX64:  a Linux host's CMake build (cmake-build/), or an injected
//                -PskainetKernelsX64Dir (CI: artifact from the ubuntu leg).
//   - linuxArm64: the -PcrossArm64 cross build (cmake-build-arm64/), or an
//                injected -PskainetKernelsArm64Dir.
// A macOS/Windows host produces host-format archives that must NOT be
// embedded into a Linux klib; those builds fall back to linkerOpts (their
// K/N test links are disabled anyway, below) and the cinterop task warns
// that the produced klib is bindings-only.
val isLinuxHostForEmbed: Boolean = System.getProperty("os.name").lowercase().contains("linux")
val crossArm64ForEmbed: Boolean = (findProperty("crossArm64") as String?)?.toBoolean() == true
val injectedX64Dir: String? = findProperty("skainetKernelsX64Dir") as String?
val injectedArm64Dir: String? = findProperty("skainetKernelsArm64Dir") as String?
val x64ArchiveDir: String? = injectedX64Dir
    ?: if (isLinuxHostForEmbed) layout.buildDirectory.dir("native/cmake-build").get().asFile.absolutePath else null
val arm64ArchiveDir: String? = injectedArm64Dir
    ?: if (crossArm64ForEmbed) layout.buildDirectory.dir("native/cmake-build-arm64").get().asFile.absolutePath else null

// Apple (K/N) archives — iOS kernel track of #920 (#959). Mach-O static
// archives built with the platform SDKs, so they exist only on a macOS host
// (Xcode required); elsewhere the whole Apple task family is disabled below
// and CI injects prebuilt archives via the -PskainetKernels*Dir properties
// (publish.yml macos leg), mirroring the Linux pair above. The device archive
// is compiled at baseline arm64 with runtime FEAT_DotProd dispatch (#958), so
// ONE archive serves A12 (no dotprod) through A18/M-series.
val isMacosHostForEmbed: Boolean =
    System.getProperty("os.name").lowercase().let { it.contains("mac") || it.contains("darwin") }
val cmakeBuildIosArm64Path: String =
    layout.buildDirectory.dir("native/cmake-build-ios-arm64").get().asFile.absolutePath
val cmakeBuildIosSimArm64Path: String =
    layout.buildDirectory.dir("native/cmake-build-ios-sim-arm64").get().asFile.absolutePath
val cmakeBuildMacosArm64Path: String =
    layout.buildDirectory.dir("native/cmake-build-macos-arm64").get().asFile.absolutePath
val injectedIosArm64Dir: String? = findProperty("skainetKernelsIosArm64Dir") as String?
val injectedIosSimArm64Dir: String? = findProperty("skainetKernelsIosSimulatorArm64Dir") as String?
val injectedMacosArm64Dir: String? = findProperty("skainetKernelsMacosArm64Dir") as String?
val iosArm64ArchiveDir: String? = injectedIosArm64Dir ?: cmakeBuildIosArm64Path.takeIf { isMacosHostForEmbed }
val iosSimArm64ArchiveDir: String? = injectedIosSimArm64Dir ?: cmakeBuildIosSimArm64Path.takeIf { isMacosHostForEmbed }
val macosArm64ArchiveDir: String? = injectedMacosArm64Dir ?: cmakeBuildMacosArm64Path.takeIf { isMacosHostForEmbed }

kotlin {
    explicitApi()
    jvm()

    // Kotlin/Native consumption of the hand-written C/NEON kernels via cinterop
    // to the static archive libskainet_kernels.a (CMake `skainet_kernels_static`).
    // linuxX64 = host (POC / CI-runnable); linuxArm64 = the SL2610 board target
    // (its archive is the aarch64 cross-build with NEON). The JVM consumes the
    // same kernels via FFM instead. Shared K/N code lives in `nativeMain`.
    fun org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget.wireSkainetKernels(
        archiveDir: String?,
        fallbackArchive: String,
    ) {
        compilations.getByName("main").cinterops.create("skainetKernels") {
            defFile(project.file("src/nativeInterop/cinterop/skainet_kernels.def"))
            includeDirs(nativeIncludeDir)
            if (archiveDir != null) {
                extraOpts("-staticLibrary", "libskainet_kernels.a", "-libraryPath", archiveDir)
            }
        }
        // Without an embeddable archive the old project-local linking keeps the
        // in-repo binaries working, but the klib itself carries no machine code.
        if (archiveDir == null) binaries.all { linkerOpts(fallbackArchive) }
    }
    linuxX64 { wireSkainetKernels(x64ArchiveDir, staticArchivePath) }
    linuxArm64 { wireSkainetKernels(arm64ArchiveDir, staticArchiveArm64Path) }
    // Apple targets (#959): the same cinterop/embedding seam, fed by the Apple
    // CMake lanes below (macOS host) or CI-injected archives.
    iosArm64 { wireSkainetKernels(iosArm64ArchiveDir, "$cmakeBuildIosArm64Path/libskainet_kernels.a") }
    iosSimulatorArm64 { wireSkainetKernels(iosSimArm64ArchiveDir, "$cmakeBuildIosSimArm64Path/libskainet_kernels.a") }
    macosArm64 { wireSkainetKernels(macosArm64ArchiveDir, "$cmakeBuildMacosArm64Path/libskainet_kernels.a") }

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
        // Shared K/N kernels (NativeKn*MatmulKernel + provider), consumed by both
        // linuxX64 and linuxArm64. The cinterop bindings are commonized across the
        // two targets so this source set can reference sk.ainet.kernels.cinterop.
        val nativeMain by creating {
            dependsOn(commonMain.get())
            dependencies {
                implementation(project(":skainet-backends:skainet-backend-api"))
            }
        }
        val linuxX64Main by getting { dependsOn(nativeMain) }
        val linuxArm64Main by getting { dependsOn(nativeMain) }
        val iosArm64Main by getting { dependsOn(nativeMain) }
        val iosSimulatorArm64Main by getting { dependsOn(nativeMain) }
        val macosArm64Main by getting { dependsOn(nativeMain) }
        // Shared K/N parity tests (cinterop kernel vs commonMain scalar reference),
        // run on BOTH linuxX64 (host, scalar/auto-vec archive) and linuxArm64
        // (cross-built NEON archive, executed under the K/N-bundled qemu-aarch64
        // or on the SL2610 board). Same tests, two codegens — this is how the
        // NEON paths get bit-checked without board-only test code.
        val nativeTest by creating {
            dependsOn(commonTest.get())
            dependencies {
                implementation(libs.kotlin.test)
                // ScalarQ*_KMatmulKernel / ScalarQ8_0MatmulKernel / ScalarMatmulKernel
                // references for the cinterop parity tests.
                implementation(project(":skainet-backends:skainet-backend-cpu"))
            }
        }
        val linuxX64Test by getting { dependsOn(nativeTest) }
        val linuxArm64Test by getting { dependsOn(nativeTest) }
        // Same parity suites on the Apple targets: iosSimulatorArm64Test /
        // macosArm64Test execute them on a macOS host (simulator / natively).
        val iosArm64Test by getting { dependsOn(nativeTest) }
        val iosSimulatorArm64Test by getting { dependsOn(nativeTest) }
        val macosArm64Test by getting { dependsOn(nativeTest) }
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

// The linuxX64 (K/N) binaries link libskainet_kernels.a (built by CMake into
// cmakeBuildPath), so the host static archive must exist before the K/N link.
tasks.matching { it.name.startsWith("link") && it.name.endsWith("LinuxX64") }.configureEach {
    dependsOn(buildNativeKernels)
}

// With archive embedding (#941) the archive must exist at CINTEROP time, not
// just link time: the cinterop task copies it into the klib.
if (x64ArchiveDir != null && injectedX64Dir == null) {
    tasks.matching { it.name == "cinteropSkainetKernelsLinuxX64" }.configureEach {
        dependsOn(buildNativeKernels)
    }
}
// Bindings-only builds warn loudly instead of publishing broken klibs silently.
if (x64ArchiveDir == null) {
    tasks.matching { it.name == "cinteropSkainetKernelsLinuxX64" }.configureEach {
        doFirst {
            logger.warn(
                "skainet-backend-native-cpu: linuxX64 klib is built WITHOUT the embedded " +
                    "libskainet_kernels.a (no Linux-ELF archive available on this host). " +
                    "Downstream consumers of this klib cannot link. Build on Linux or pass " +
                    "-PskainetKernelsX64Dir=<dir> (#941)."
            )
        }
    }
}
if (arm64ArchiveDir == null) {
    tasks.matching { it.name == "cinteropSkainetKernelsLinuxArm64" }.configureEach {
        doFirst {
            logger.warn(
                "skainet-backend-native-cpu: linuxArm64 klib is built WITHOUT the embedded " +
                    "libskainet_kernels.a. Downstream consumers of this klib cannot link. " +
                    "Enable -PcrossArm64=true (aarch64 cross toolchain) or pass " +
                    "-PskainetKernelsArm64Dir=<dir> (#941)."
            )
        }
    }
}

// The linuxX64/linuxArm64 K/N *test* binaries link the CMake static archive and
// execute a Linux ELF. On a non-Linux host the CMake build emits host-format
// objects (Mach-O on macOS), which ld.lld cannot cross-link into a Linux binary,
// and a Linux .kexe cannot be executed anyway. Disable the K/N test link + run
// on non-Linux hosts so `build`/`allTests` stay green locally; klib compilation
// (compileKotlinLinux*) and publishing are unaffected, and the native NEON parity
// suite still runs on Linux CI / qemu-aarch64 / the SL2610 board. Mirrors the
// existing `-PcrossArm64` opt-in gating for the aarch64 cross artifacts.
val isLinuxHost: Boolean = System.getProperty("os.name").lowercase().contains("linux")
if (!isLinuxHost) {
    tasks.matching {
        (it.name.startsWith("link") && it.name.contains("Test") &&
            (it.name.endsWith("LinuxX64") || it.name.endsWith("LinuxArm64"))) ||
            it.name == "linuxX64Test" || it.name == "linuxArm64Test"
    }.configureEach { enabled = false }
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
// NEON parity verified under qemu-aarch64 (see skainet_simd.h banner):
//   ./gradlew :skainet-backends:skainet-backend-native-cpu:linuxArm64Test -PcrossArm64=true
// runs the shared nativeTest parity suite against the cross-built archive.
val crossArm64Enabled: Boolean = (findProperty("crossArm64") as String?)?.toBoolean() == true
val aarch64Cc: String = (findProperty("skainetAarch64Cc") as String?) ?: "aarch64-linux-gnu-gcc"
val cmakeBuildArm64Path: String = layout.buildDirectory.dir("native/cmake-build-arm64").get().asFile.absolutePath
val nativeResourceArm64Dir = nativeResourcesRoot.map { it.dir("native/linux-arm64") }
val toolchainFilePath = "$nativeSourcePath/toolchain-aarch64.cmake"

val configureNativeKernelsArm64 by tasks.registering(Exec::class) {
    group = "build"
    description = "CMake configure for the aarch64 (NEON) native kernels (cross-compile)."
    // Local copy so the onlyIf lambda captures a Boolean, not the build script
    // (script object references break configuration-cache serialization).
    val enabled = crossArm64Enabled
    onlyIf { enabled }
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
    val enabled = crossArm64Enabled
    onlyIf { enabled }
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
    val enabled = crossArm64Enabled
    onlyIf { enabled }
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

// linuxArm64 binaries link the aarch64 cross-built archive. Only wired with
// -PcrossArm64 (cross toolchain present): a plain host build still compiles
// linuxArm64 to a klib (no archive needed) — only a final binary/test link
// needs it, which is a board/CI concern.
if (crossArm64Enabled) {
    tasks.matching { it.name.startsWith("link") && it.name.endsWith("LinuxArm64") }.configureEach {
        dependsOn(buildNativeKernelsArm64)
    }

    // Embedding (#941): the cross-built archive must exist before the arm64
    // cinterop copies it into the klib.
    if (injectedArm64Dir == null) {
        tasks.matching { it.name == "cinteropSkainetKernelsLinuxArm64" }.configureEach {
            dependsOn(buildNativeKernelsArm64)
        }
    }

    // The Kotlin Gradle plugin does not create a run task for non-host K/N test
    // binaries (only linkDebugTestLinuxArm64 / linuxArm64TestBinaries), so wire
    // one explicitly: run test.kexe under qemu-aarch64 (user-mode emulation).
    // Defaults point at the K/N-bundled dependencies (~/.konan/dependencies):
    // the same qemu K/N itself uses for linux_x64 -> linux_arm64 test emulation
    // (konan.properties emulatorExecutable.linux_x64-linux_arm64) and the glibc
    // sysroot the binary was linked against. Override with -PskainetQemu /
    // -PskainetAarch64Sysroot (e.g. /usr/bin/qemu-aarch64-static and
    // /usr/aarch64-linux-gnu for the distro toolchain). On the SL2610 board
    // itself, just push and run test.kexe directly — no qemu involved.
    val konanDeps = "${System.getProperty("user.home")}/.konan/dependencies"
    val qemuAarch64: String = (findProperty("skainetQemu") as String?)
        ?: "$konanDeps/qemu-aarch64-static-5.1.0-linux-2/qemu-aarch64"
    val aarch64Sysroot: String = (findProperty("skainetAarch64Sysroot") as String?)
        ?: "$konanDeps/aarch64-unknown-linux-gnu-gcc-8.3.0-glibc-2.25-kernel-4.9-2/aarch64-unknown-linux-gnu/sysroot"
    val testKexePath: String =
        layout.buildDirectory.file("bin/linuxArm64/debugTest/test.kexe").get().asFile.absolutePath

    tasks.register<Exec>("linuxArm64Test") {
        group = "verification"
        description = "Run the linuxArm64 K/N tests under qemu-aarch64 (NEON kernel parity vs scalar reference)."
        dependsOn("linkDebugTestLinuxArm64")
        inputs.file(testKexePath)
        commandLine(qemuAarch64, "-L", aarch64Sysroot, testKexePath)
    }
}

// --- Apple archives (#959, iOS kernel track of #920) ------------------------
//
// Three Mach-O static archives for the Apple K/N targets, built with the
// platform SDKs on a macOS host (CMake >= 3.14 has first-class iOS support —
// no toolchain file needed; contrast toolchain-aarch64.cmake, which exists
// for the Linux cross triple). No -march is passed: Apple builds stay at the
// SDK-default baseline and rely on the #958 runtime FEAT_DotProd dispatch.
// Deployment targets match the K/N binary minimums (iOS 12.0 / simulator
// 14.0 / macOS 11.0) so the K/N link never warns about newer object files.
fun registerAppleKernelTasks(
    suffix: String,
    buildDir: String,
    extraCmakeArgs: List<String>,
): TaskProvider<Exec> {
    // Local copy: the onlyIf lambda must capture a Boolean, not the script.
    val macHost = isMacosHostForEmbed
    val configure = tasks.register<Exec>("configureNativeKernels$suffix") {
        group = "build"
        description = "CMake configure for the $suffix (Apple) native kernels."
        onlyIf { macHost }
        inputs.file("$nativeSourcePath/CMakeLists.txt")
        inputs.dir("$nativeSourcePath/src")
        inputs.dir("$nativeSourcePath/include")
        outputs.dir(buildDir)
        commandLine = listOf(
            "cmake", "-S", nativeSourcePath, "-B", buildDir,
            "-DCMAKE_BUILD_TYPE=Release",
        ) + extraCmakeArgs
    }
    return tasks.register<Exec>("buildNativeKernels$suffix") {
        group = "build"
        description = "Build the $suffix (Apple) native kernels static archive."
        onlyIf { macHost }
        dependsOn(configure)
        inputs.file("$nativeSourcePath/CMakeLists.txt")
        inputs.dir("$nativeSourcePath/src")
        inputs.dir("$nativeSourcePath/include")
        outputs.dir(buildDir)
        commandLine = listOf("cmake", "--build", buildDir, "--config", "Release")
    }
}

val buildNativeKernelsIosArm64 = registerAppleKernelTasks(
    "IosArm64", cmakeBuildIosArm64Path,
    listOf(
        "-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_SYSROOT=iphoneos",
        "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
    ),
)
val buildNativeKernelsIosSimulatorArm64 = registerAppleKernelTasks(
    "IosSimulatorArm64", cmakeBuildIosSimArm64Path,
    listOf(
        "-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_SYSROOT=iphonesimulator",
        "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
    ),
)
val buildNativeKernelsMacosArm64 = registerAppleKernelTasks(
    "MacosArm64", cmakeBuildMacosArm64Path,
    listOf(
        "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0",
        // The macOS lane exists for the K/N klib; the FFM dylib comes from the
        // regular host build (packageNativeKernels) on the macos CI leg.
        "-DSKAINET_STATIC_ONLY=ON",
    ),
)

// Embedding (#941 semantics): the archive must exist at CINTEROP time (it is
// copied into the klib) and at this project's own test-binary link time;
// bindings-only builds warn loudly instead of publishing broken klibs silently.
fun wireAppleCinterop(
    suffix: String,
    archiveDir: String?,
    injected: String?,
    buildTask: TaskProvider<Exec>,
    propName: String,
) {
    if (archiveDir != null && injected == null) {
        tasks.matching { it.name == "cinteropSkainetKernels$suffix" }.configureEach { dependsOn(buildTask) }
        tasks.matching { it.name.startsWith("link") && it.name.endsWith(suffix) }.configureEach { dependsOn(buildTask) }
    }
    if (archiveDir == null) {
        tasks.matching { it.name == "cinteropSkainetKernels$suffix" }.configureEach {
            doFirst {
                logger.warn(
                    "skainet-backend-native-cpu: ${suffix.replaceFirstChar { it.lowercase() }} klib is built " +
                        "WITHOUT the embedded libskainet_kernels.a (no Apple archive available on this " +
                        "host). Downstream consumers of this klib cannot link. Build on macOS or pass " +
                        "-P$propName=<dir> (#959)."
                )
            }
        }
    }
}
wireAppleCinterop("IosArm64", iosArm64ArchiveDir, injectedIosArm64Dir, buildNativeKernelsIosArm64, "skainetKernelsIosArm64Dir")
wireAppleCinterop("IosSimulatorArm64", iosSimArm64ArchiveDir, injectedIosSimArm64Dir, buildNativeKernelsIosSimulatorArm64, "skainetKernelsIosSimulatorArm64Dir")
wireAppleCinterop("MacosArm64", macosArm64ArchiveDir, injectedMacosArm64Dir, buildNativeKernelsMacosArm64, "skainetKernelsMacosArm64Dir")

// The Apple task family needs macOS (Xcode SDKs), and klib cross-compilation
// of cinterop-bearing modules is not supported off-macOS anyway. Disable the
// whole family elsewhere so ubuntu CI stays deterministic regardless of KGP
// defaults; publishing runs on macOS (publish.yml). Mirrors the non-Linux
// gate above for the Linux K/N tests.
if (!isMacosHostForEmbed) {
    tasks.matching { t ->
        listOf("IosArm64", "IosSimulatorArm64", "MacosArm64").any { t.name.endsWith(it) } ||
            t.name in setOf("iosArm64Test", "iosSimulatorArm64Test", "macosArm64Test")
    }.configureEach { enabled = false }
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
