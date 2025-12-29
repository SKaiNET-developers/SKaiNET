import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.ksp)
    alias(libs.plugins.dokka)
}

kotlin {
    explicitApi()

    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    sourceSets {
        commonMain.dependencies {
            api(project(":skainet-lang:skainet-lang-ksp-annotations"))
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

dependencies {
    // KSP processor for generating tracing wrappers
    add("kspCommonMainMetadata", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspJvm", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspAndroid", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspIosArm64", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspIosSimulatorArm64", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspMacosArm64", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspLinuxX64", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspLinuxArm64", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspJs", project(":skainet-lang:skainet-lang-ksp-processor"))
    add("kspWasmJs", project(":skainet-lang:skainet-lang-ksp-processor"))
}

android {
    namespace = "sk.ainet.lang.core"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

tasks.named("dokkaHtml") {
    dependsOn("kspCommonMainKotlinMetadata")
}