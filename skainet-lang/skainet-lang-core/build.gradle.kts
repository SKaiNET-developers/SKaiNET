import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.ksp)
    id("sk.ainet.dokka")
    id("org.jetbrains.kotlinx.benchmark")
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.lang.core"
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

    // Android Native targets for vendor-specific backends linking native device libs.
    androidNativeArm32()
    androidNativeArm64()

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
        commonMain {
            // Include KSP-generated sources in commonMain so downstream modules can access them
            kotlin.srcDir("build/generated/ksp/metadata/commonMain/kotlin")
            dependencies {
                api(project(":skainet-lang:skainet-lang-ksp-annotations"))
            }
        }

        jvmMain.dependencies {
            implementation(libs.kotlinx.benchmark.runtime)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

// Ensure KSP metadata task runs before any Kotlin compilation, other KSP tasks, and sourcesJar
tasks.configureEach {
    if (name != "kspCommonMainKotlinMetadata" &&
        (name.startsWith("compileKotlin") || name.startsWith("ksp") || name.contains("ourcesJar"))) {
        dependsOn("kspCommonMainKotlinMetadata")
    }
}

dependencies {
    // KSP processor for generating tracing wrappers - only for commonMain metadata
    // to avoid duplicate class generation across targets
    add("kspCommonMainMetadata", project(":skainet-lang:skainet-lang-ksp-processor"))
}

// Pass the canonical SKaiNET version (VERSION_NAME in the root
// gradle.properties — same value that's published to Maven Central) to
// the OperatorDocProcessor so generated ops pages stamp a real version
// instead of the hardcoded "1.0.0" placeholder. Read at configuration
// time via providers.gradleProperty so build-cache entries invalidate
// when the version bumps.
ksp {
    arg("skainet.version", providers.gradleProperty("VERSION_NAME").getOrElse("unknown"))
}

tasks.matching { it.name.startsWith("dokka") }.configureEach {
    dependsOn("kspCommonMainKotlinMetadata")
}

benchmark {
    targets {
        register("jvm")
    }
}