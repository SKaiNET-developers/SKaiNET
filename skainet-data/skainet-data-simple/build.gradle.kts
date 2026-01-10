import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
}

kotlin {
    explicitApi()

    targets.configureEach {
        compilations.configureEach {
            compileTaskProvider.get().compilerOptions {
                freeCompilerArgs.add("-Xexpect-actual-classes")
            }
        }
    }

    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()

    macosArm64()

    linuxX64()
    linuxArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    sourceSets {
        androidMain.dependencies {
            implementation(libs.ktor.client.android)
        }
        val commonMain by getting {
            dependencies {
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(project(":skainet-data:skainet-data-api"))
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.ktor.client.core)
                implementation(libs.kotlinx.coroutines)
            }
        }

        val iosMain by creating {
            dependsOn(commonMain)
            dependencies {
                implementation(libs.ktor.client.darwin)
            }
        }
        val iosArm64Main by getting {
            dependsOn(iosMain)
        }
        val iosSimulatorArm64Main by getting {
            dependsOn(iosMain)
        }

        val macosMain by creating {
            dependsOn(commonMain)
        }
        val macosArm64Main by getting {
            dependsOn(macosMain)
        }

        val linuxMain by creating {
            dependsOn(commonMain)
        }
        val linuxX64Main by getting {
            dependsOn(linuxMain)
        }
        val linuxArm64Main by getting {
            dependsOn(linuxMain)
        }

        jvmMain.dependencies {
            implementation(libs.ktor.client.cio)
            implementation(libs.ktor.client.plugins)
            implementation(libs.ktor.client.logging)
            implementation(libs.ktor.client.content.negotiation)
            implementation(libs.kotlinx.coroutines.core.jvm)
            implementation(libs.logback.classic) // For logging
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        val jsMain by getting {
            dependencies {
                implementation(libs.ktor.client.js)
            }
        }
        val wasmJsMain by getting {
            dependencies {
                implementation(libs.ktor.client.js)
            }
        }
    }
}

android {
    namespace = "sk.ainet.core.api"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}