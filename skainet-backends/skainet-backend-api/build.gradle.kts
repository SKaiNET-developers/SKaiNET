import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()
    android {
        namespace = "sk.ainet.backend.api"
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
    androidNativeArm32()

    jvm()

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
        commonMain.dependencies {
            // Neutral backend API is an `api` re-export of the tensor op and
            // storage interfaces already defined in skainet-lang-core. Any
            // concrete backend (CPU, IREE, Metal, NPU, ...) should depend on
            // this module instead of pulling in skainet-backend-cpu just to
            // reach TensorOps / TensorDataFactory / TensorData.
            api(project(":skainet-lang:skainet-lang-core"))
        }

        val jvmMain by getting
        val androidMain by getting
        val wasmJsMain by getting

        val commonMain by getting

        val nativeMain by creating {
            dependsOn(commonMain)
        }

        val appleMain by creating {
            dependsOn(nativeMain)
        }

        val linuxMain by creating {
            dependsOn(nativeMain)
        }

        val iosMain by creating {
            dependsOn(appleMain)
        }

        val macosMain by creating {
            dependsOn(appleMain)
        }

        val iosArm64Main by getting {
            dependsOn(iosMain)
        }

        val iosSimulatorArm64Main by getting {
            dependsOn(iosMain)
        }

        val macosArm64Main by getting {
            dependsOn(macosMain)
        }

        val linuxX64Main by getting {
            dependsOn(linuxMain)
        }

        val linuxArm64Main by getting {
            dependsOn(linuxMain)
        }

        val androidNativeArm32Main by getting {
            dependsOn(nativeMain)
        }
    }
}
