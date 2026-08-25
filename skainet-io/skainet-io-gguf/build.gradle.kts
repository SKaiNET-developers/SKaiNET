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

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlinx.io.core)
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(project(":skainet-io:skainet-io-core"))
                implementation(project(":skainet-compile:skainet-compile-core"))
                implementation(project(":skainet-compile:skainet-compile-dag"))

            }
        }
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
                // #1118's acceptance test loads a model and *runs* it, so it needs a backend that
                // computes: this module's own DefaultDataExecutionContext carries VoidTensorOps.
                // Test-only, and not a cycle — the CPU backend does not know about GGUF.
                implementation(project(":skainet-backends:skainet-backend-cpu"))
                implementation(project(":skainet-backends:skainet-backend-api"))
            }
        }

        // Host-side tests of the *Android* compilation (#1038): they drive the suspending loader,
        // so they need coroutines like jvmTest does. The GGUF fixtures they write are their own —
        // jvmTest's SyntheticGguf is not visible from this compilation.
        getByName("androidHostTest") {
            dependencies {
                implementation(libs.kotlinx.coroutines)
            }
        }
    }
}
