import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

kotlin {
    targets.configureEach {
        compilations.configureEach {
            compileTaskProvider.get().compilerOptions {
                freeCompilerArgs.add("-Xexpect-actual-classes")
            }
        }
    }

    jvm()
    android {
        namespace = "sk.ainet.io.gguf"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
        // Host-side (JVM) unit tests for the Android compilation — exercises
        // the mmap-backed weight path (MappedGgufWeights) without a device (#921).
        withHostTest {}
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()
    // 32-bit: RandomAccessSourceFactory needs its own actual (see
    // androidNativeArm32Main) — pread's ssize_t/off_t are Int here vs Long on
    // every other native target, so it can't share nativeMain's actual with
    // them (see the native64Main split below and skainet-io-core's own).
    androidNativeArm32()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    // 64-bit-only intermediate: holds the pread-based RandomAccessSourceFactory
    // actual, whose ssize_t/off_t widths are uniform (Long) across every native
    // target EXCEPT androidNativeArm32 (which gets its own actual instead — see
    // androidNativeArm32Main). Mirrors skainet-io-core's identical split.
    val native64Targets = listOf("iosArm64", "iosSimulatorArm64", "macosArm64", "linuxX64", "linuxArm64")
    applyDefaultHierarchyTemplate()

    sourceSets {
        val nativeMain by getting
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlinx.io.core)
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(project(":skainet-io:skainet-io-core"))
            }
        }
        // GgufExportFacade (host-side model export/compile tooling) needs
        // skainet-compile-dag, which doesn't build for androidNativeArm32
        // (32-bit) — see PosixPreadRandomAccessSource's own split for the
        // identical reasoning. Model export isn't an on-device concern anyway,
        // so it lives here instead of commonMain: jvm/android/native64 (every
        // target except androidNativeArm32) get it.
        val exportMain by creating {
            dependsOn(commonMain)
            dependencies {
                implementation(project(":skainet-compile:skainet-compile-core"))
                implementation(project(":skainet-compile:skainet-compile-dag"))
            }
        }
        val native64Main by creating {
            dependsOn(nativeMain)
            dependsOn(exportMain)
        }
        native64Targets.forEach { t -> getByName("${t}Main").dependsOn(native64Main) }
        val jvmMain by getting { dependsOn(exportMain) }
        val androidMain by getting { dependsOn(exportMain) }
        // js/wasmJs/wasmWasi: skainet-compile-dag supports these too — only
        // androidNativeArm32 is actually excluded from exportMain.
        val jsMain by getting { dependsOn(exportMain) }
        val wasmJsMain by getting { dependsOn(exportMain) }
        val wasmWasiMain by getting { dependsOn(exportMain) }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        // Memory-mapped GGUF weight access (MappedGgufWeights), shared source
        // between the JVM and Android compilations (#921): pure java.nio
        // (FileChannel.map — API 1 on Android). Shared directory rather than an
        // intermediate source set — each compilation builds against its full
        // platform classpath.
        getByName("jvmMain") {
            kotlin.srcDir("src/jvmAndroidMain/kotlin")
        }
        getByName("androidMain") {
            kotlin.srcDir("src/jvmAndroidMain/kotlin")
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.junit)
                implementation(libs.kotlinx.coroutines)
                implementation(libs.kotlinx.coroutines.test)
            }
        }
    }
}
