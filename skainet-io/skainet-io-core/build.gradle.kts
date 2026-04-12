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

tasks.withType<Test>().configureEach {
    systemProperty("skainet.test.fixturesDir", fixturesDir.get().asFile.absolutePath)
}
