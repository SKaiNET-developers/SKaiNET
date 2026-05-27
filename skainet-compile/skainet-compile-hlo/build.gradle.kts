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
        namespace = "sk.ainet.compile.hlo"
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
            api(project(":skainet-compile:skainet-compile-core"))
            api(project(":skainet-compile:skainet-compile-dag"))
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-backends:skainet-backend-cpu"))

        }

        jvmMain.dependencies {
            // HloGenerator records traces with VoidTensorOps from
            // skainet-lang-core — the JVM production path never needs a
            // real backend implementation. No CPU-specific imports here.
            implementation(project(":skainet-lang:skainet-lang-models"))
            implementation(libs.kotlinx.coroutines)
        }

        jvmTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}

// CLI task to generate StableHLO MLIR from registered sample models.
// Usage:
//   ./gradlew :skainet-compile:skainet-compile-hlo:generateHlo -Pmodel=list
//   ./gradlew :skainet-compile:skainet-compile-hlo:generateHlo -Pmodel=rgb2grayscale
//   ./gradlew :skainet-compile:skainet-compile-hlo:generateHlo \
//       -Pmodel=rgb2grayscale -Poutput=rgb2grayscale.mlir -Pheight=224 -Pwidth=224
val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")

tasks.register<JavaExec>("generateHlo") {
    group = "application"
    description = "Generates StableHLO MLIR from a registered sample model."

    dependsOn(tasks.named("jvmJar"))

    mainClass.set("sk.ainet.compile.hlo.generate.HloGeneratorMainKt")

    classpath = files(
        jvmMainCompilation.runtimeDependencyFiles,
        tasks.named("jvmJar").get().outputs.files
    )

    val argsList = mutableListOf<String>()
    providers.gradleProperty("model").orNull?.let { argsList += "--model=$it" }
    providers.gradleProperty("output").orNull?.let { argsList += "--output=$it" }
    providers.gradleProperty("height").orNull?.let { argsList += "--height=$it" }
    providers.gradleProperty("width").orNull?.let { argsList += "--width=$it" }
    providers.gradleProperty("batch").orNull?.let { argsList += "--batch=$it" }
    args = argsList
}
