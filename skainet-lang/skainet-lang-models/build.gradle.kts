import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

kotlin {

    explicitApi()

    android {
        namespace = "sk.ainet.lang.models"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
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
        browser {
            testTask {
                useMocha {
                    timeout = "10s"
                }
            }
        }
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser {
            testTask {
                useMocha {
                    timeout = "10s"
                }
            }
        }
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-compile:skainet-compile-dag"))
            implementation(project(":skainet-compile:skainet-compile-json"))

        }
    }
}