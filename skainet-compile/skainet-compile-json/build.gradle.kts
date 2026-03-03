import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kotlinSerialization)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.compilie.json"
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
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-compile:skainet-compile-dag"))
            implementation(libs.kotlinx.serialization.json)
            implementation(libs.kotlinx.coroutines)

        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        jvmTest.dependencies {
            // Use DSL example models and a simple CPU execution context for integration tests
            implementation(project(":skainet-lang:skainet-lang-models"))
            implementation(project(":skainet-backends:skainet-backend-cpu"))
        }
    }
}

// Minimal CLI task to run the JSON export proof of concept
// Usage example:
//   ./gradlew :skainet-compile:skainet-compile-json:exportJson \
//       -Poutput=build/exports/tiny.json -Plabel=tiny_graph
val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")

tasks.register<JavaExec>("exportJson") {
    group = "application"
    description = "Exports a tiny synthetic graph to JSON (proof of concept)."

    // Ensure the JAR is built before running
    dependsOn(tasks.named("jvmJar"))

    mainClass.set("sk.ainet.compile.json.MainKt")

    // Compose classpath from runtime deps + the compiled jar
    classpath = files(
        jvmMainCompilation.runtimeDependencyFiles,
        tasks.named("jvmJar").get().outputs.files
    )

    // Forward CLI parameters as system properties/args
    // Supported project properties: output, label
    val out = (project.findProperty("output") as String?)
    val lbl = (project.findProperty("label") as String?)
    val argsList = mutableListOf<String>()
    if (out != null) argsList += "--output=$out"
    if (lbl != null) argsList += "--label=$lbl"
    args = argsList
}
