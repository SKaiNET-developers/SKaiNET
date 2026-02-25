@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.kotlinSerialization)

    alias(libs.plugins.vanniktech.mavenPublish)
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

    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
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

android {
    namespace = "sk.ainet.io.core"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
}