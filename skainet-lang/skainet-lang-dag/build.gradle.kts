import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.ksp)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.lang.dag"
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
            kotlin.srcDir("build/generated/ksp/metadata/commonMain/kotlin")
            dependencies {
                api(project(":skainet-lang:skainet-lang-core"))
                implementation(libs.kotlinx.coroutines)
            }
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-backends:skainet-backend-cpu"))
        }
    }
}

// Ensure KSP metadata task runs before any Kotlin compilation, other KSP tasks, and sourcesJar
tasks.configureEach {
    if (name != "kspCommonMainKotlinMetadata" &&
        (name.startsWith("compileKotlin") || name.startsWith("ksp") || name.contains("ourcesJar"))
    ) {
        dependsOn("kspCommonMainKotlinMetadata")
    }
}

tasks.matching { it.name.startsWith("dokka") }.configureEach {
    dependsOn("kspCommonMainKotlinMetadata")
}

dependencies {
    add("kspCommonMainMetadata", project(":skainet-lang:skainet-lang-ksp-processor"))
}
