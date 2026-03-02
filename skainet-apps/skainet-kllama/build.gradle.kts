import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    linuxX64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    linuxArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-apps:skainet-kllama-agent"))
            implementation(project(":skainet-apps:skainet-llm"))
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
            implementation(project(":skainet-io:skainet-io-core"))
            implementation(project(":skainet-io:skainet-io-gguf"))
            implementation(project(":skainet-io:skainet-io-safetensors"))
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.serialization.json)

        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-lang:skainet-lang-models"))
            implementation(project(":skainet-io:skainet-io-gguf"))
        }

        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
                implementation(project(":skainet-backends:skainet-backend-cpu"))
            }
        }
        // val androidMain by getting
        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting
        }
        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }

        val nativeMain by creating {
            dependsOn(commonMain.get())
        }
        val linuxMain by creating { dependsOn(nativeMain) }
        val macosMain by creating { dependsOn(nativeMain) }
        val linuxX64Main by getting { dependsOn(linuxMain) }
        val linuxArm64Main by getting { dependsOn(linuxMain) }
        val macosArm64Main by getting { dependsOn(macosMain) }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
    maxHeapSize = "6g"
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}


android {
    namespace = "sk.ainet.apps.kllama"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}