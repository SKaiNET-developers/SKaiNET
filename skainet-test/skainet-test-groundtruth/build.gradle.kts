import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
}

kotlin {
    explicitApi()
    jvmToolchain(21)

    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm()

    sourceSets {
        commonMain.dependencies {
            // Core tensor and operations
            api(project(":skainet-lang:skainet-lang-core"))

            // GGUF file reading
            api(project(":skainet-io:skainet-io-gguf"))

            // CPU backend for executing operations
            api(project(":skainet-backends:skainet-backend-cpu"))

            // IO utilities
            implementation(libs.kotlinx.io.core)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        jvmMain.dependencies {
            // JVM-specific dependencies if needed
        }

        jvmTest.dependencies {
            implementation(libs.junit)
        }
    }
}

android {
    namespace = "sk.ainet.test.groundtruth"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
