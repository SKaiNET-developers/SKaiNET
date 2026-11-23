plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
    alias(libs.plugins.ksp)
}

skainet {
    namespace = "sk.ainet.backend.cpu"
}

kotlin {
    androidLibrary {
        // Run commonTest as host-side (JVM) unit tests of the Android compilation (#1230).
        withHostTest {}
    }

    sourceSets {
        commonMain.dependencies {
            // Every concrete backend should go through the neutral api
            // module; it transitively brings in skainet-lang-core.
            implementation(project(":skainet-backends:skainet-backend-api"))
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
        }

        commonTest.dependencies {
            implementation(project(":skainet-lang:skainet-lang-models"))
        }

        jvmMain.dependencies {
            implementation(libs.kotlinx.coroutines)
        }

        // This module opts out of the default hierarchy template
        // (kotlin.mpp.applyDefaultHierarchyTemplate=false in gradle.properties),
        // so the native tree is wired by hand.
        nativeMain { dependsOn(commonMain.get()) }
        appleMain { dependsOn(nativeMain.get()) }
        linuxMain { dependsOn(nativeMain.get()) }
        iosMain { dependsOn(appleMain.get()) }
        macosMain { dependsOn(appleMain.get()) }

        iosArm64Main { dependsOn(iosMain.get()) }
        iosSimulatorArm64Main { dependsOn(iosMain.get()) }
        macosArm64Main { dependsOn(macosMain.get()) }
        linuxX64Main { dependsOn(linuxMain.get()) }
        linuxArm64Main { dependsOn(linuxMain.get()) }

        // See PlatformCpuOpsFactory.androidNative.kt for why these hang off nativeMain and
        // not linuxMain.
        androidNativeMain { dependsOn(nativeMain.get()) }
        androidNativeArm32Main { dependsOn(androidNativeMain.get()) }
        androidNativeArm64Main { dependsOn(androidNativeMain.get()) }

        // Golden (bit-identical) regression tests for the packed encodings — SKEEP-003 gate.
        // Shared by the JVM and Kotlin/Native test targets only: JS/Wasm compute Float in
        // double precision and are not expected to be bit-identical (they are covered by the
        // tolerance-based parity tests in commonTest).
        val goldenTestDir = "src/goldenTest/kotlin"
        jvmTest { kotlin.srcDir(goldenTestDir) }
        linuxX64Test { kotlin.srcDir(goldenTestDir) }
        linuxArm64Test { kotlin.srcDir(goldenTestDir) }
        macosArm64Test { kotlin.srcDir(goldenTestDir) }
    }
}

// Run the operator-doc KSP processor over commonMain so the coverage matrix gets a real
// `cpu` column: it sees `TensorOps` (from skainet-lang-core) plus the `@Backend("cpu")`
// `DefaultCpuOps` overrides and their @NotImplemented / @InProgress markers. Root
// `generateDocs` merges the emitted operators.json via `documentation.backendStatusFiles`.
dependencies {
    add("kspCommonMainMetadata", project(":skainet-lang:skainet-lang-ksp-processor"))
}

ksp {
    arg("skainet.version", providers.gradleProperty("VERSION_NAME").getOrElse("unknown"))
    arg("skainet.module", "skainet-backend-cpu")
}

// Forward the BLAS switches from the Gradle CLI to the forked test JVM, so a suite can be run against
// the platform BLAS (`-Dskainet.cpu.blas.enabled=true`, optionally `-Dskainet.cpu.blas.library=<path>`).
val blasEnabledProperty = providers.systemProperty("skainet.cpu.blas.enabled")
val blasLibraryProperty = providers.systemProperty("skainet.cpu.blas.library")

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
    blasEnabledProperty.orNull?.let { systemProperty("skainet.cpu.blas.enabled", it) }
    blasLibraryProperty.orNull?.let { systemProperty("skainet.cpu.blas.library", it) }
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
