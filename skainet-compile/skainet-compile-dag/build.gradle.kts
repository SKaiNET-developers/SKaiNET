import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.compilie.dag"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()
    androidNativeArm32()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            api(project(":skainet-lang:skainet-lang-core"))
            api(project(":skainet-lang:skainet-lang-dag"))
            api(project(":skainet-compile:skainet-compile-core"))
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-lang:skainet-lang-models"))

        }
    }
}
