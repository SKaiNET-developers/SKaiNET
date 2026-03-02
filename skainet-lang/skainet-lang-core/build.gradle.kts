import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.ksp)
    alias(libs.plugins.dokka)
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

tasks.named("dokkaHtml") {
    dependsOn("kspCommonMainKotlinMetadata")
}

tasks.named("dokkaJavadoc") {
    dependsOn("kspCommonMainKotlinMetadata")
}

benchmark {
    targets {
        register("jvm")
    }
}