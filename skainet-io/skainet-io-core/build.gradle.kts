@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.net.URI

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.kotlinSerialization)

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
        namespace = "sk.ainet.io.core"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_1_8)
        }
        // Host-side (JVM) unit tests for androidMain — exercises
        // AndroidRandomAccessSource without a device (#922).
        withHostTest {}
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()
    androidNativeArm64()
    // androidNativeArm32 is 32-bit: posix ssize_t/size_t are Int here vs Long on every other
    // native target. PosixPreadRandomAccessSource therefore lives in the 64-bit-only `native64Main`
    // source set (wired below), NOT in the shared `nativeMain` — otherwise the shared native
    // metadata compile fails ("numbers with different bit widths"). arm32 gets the rest of
    // io-core (tokenizers etc.); on-device file I/O for arm32 is a separate concern.
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

    // 64-bit-only intermediate source sets: hold the posix `pread` RandomAccessSource, whose
    // ssize_t/size_t widths are uniform (Long) across every native target EXCEPT androidNativeArm32.
    // Keeping it out of the all-native `nativeMain` avoids the mixed-width metadata compile error.
    applyDefaultHierarchyTemplate()
    val native64Targets = listOf("iosArm64", "iosSimulatorArm64", "macosArm64", "linuxX64", "linuxArm64", "androidNativeArm64")

    sourceSets {
        val nativeMain by getting
        val nativeTest by getting
        val native64Main by creating { dependsOn(nativeMain) }
        val native64Test by creating { dependsOn(nativeTest) }
        native64Targets.forEach { t ->
            getByName("${t}Main").dependsOn(native64Main)
            getByName("${t}Test").dependsOn(native64Test)
        }

        val commonMain by getting {
            dependencies {
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(libs.kotlinx.io.core)
                implementation(libs.kotlinx.coroutines)

                implementation(libs.kotlinx.serialization.json)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
                implementation(project(":skainet-backends:skainet-backend-cpu"))
                implementation(project(":skainet-io:skainet-io-gguf"))
            }
        }

        val jsMain by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
            }
        }

        val wasmJsMain by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
                implementation(libs.kotlinx.browser)
            }
        }

        val wasmJsTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines.test)
            }
        }
    }
}

// ============================================================================
// Test fixtures for QwenByteLevelBpeTokenizer end-to-end tests (#463).
//
// Downloads a small public Qwen2.5 model + tokenizer.json into
// build/test-fixtures/. Tests check for file presence and skip cleanly
// when absent, so offline/CI builds without network still stay green.
//
// Run `./gradlew :skainet-io:skainet-io-core:downloadQwenTokenizerFixtures`
// once before running the fixture-gated tests.
// ============================================================================
val fixturesDir = layout.buildDirectory.dir("test-fixtures")

val downloadQwenTokenizerFixtures by tasks.registering {
    group = "verification"
    description = "Download Qwen2.5-0.5B GGUF + tokenizer.json for #463 tests"
    val outDir = fixturesDir
    outputs.dir(outDir)
    doLast {
        val dir = outDir.get().asFile.apply { mkdirs() }
        val files = listOf(
            "Qwen2.5-0.5B-Instruct-Q8_0.gguf" to
                "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
            "tokenizer.json" to
                "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
        )
        for ((name, url) in files) {
            val target = dir.resolve(name)
            if (target.exists() && target.length() > 0) {
                logger.lifecycle("fixture already present: ${target.name}")
                continue
            }
            logger.lifecycle("downloading $name from $url")
            URI(url).toURL().openStream().use { input ->
                target.outputStream().use { out -> input.copyTo(out) }
            }
            logger.lifecycle("  -> ${target.length()} bytes")
        }
    }
}

val downloadTinyLlamaTokenizerFixtures by tasks.registering {
    group = "verification"
    description = "Download TinyLlama-1.1B GGUF + tokenizer.json for #464 tests"
    val outDir = fixturesDir
    outputs.dir(outDir)
    doLast {
        val dir = outDir.get().asFile.apply { mkdirs() }
        val files = listOf(
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf" to
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            "tinyllama-tokenizer.json" to
                "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json",
        )
        for ((name, url) in files) {
            val target = dir.resolve(name)
            if (target.exists() && target.length() > 0) {
                logger.lifecycle("fixture already present: ${target.name}")
                continue
            }
            logger.lifecycle("downloading $name from $url")
            URI(url).toURL().openStream().use { input ->
                target.outputStream().use { out -> input.copyTo(out) }
            }
            logger.lifecycle("  -> ${target.length()} bytes")
        }
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("skainet.test.fixturesDir", fixturesDir.get().asFile.absolutePath)
}
