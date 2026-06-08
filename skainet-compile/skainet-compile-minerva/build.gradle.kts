import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

kotlin {
    explicitApi()

    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }

    sourceSets {
        commonMain.dependencies {
            api(project(":skainet-lang:skainet-lang-core"))
            api(project(":skainet-compile:skainet-compile-core"))
            api(project(":skainet-compile:skainet-compile-dag"))
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

val minervaHostVerificationEnabled = providers.gradleProperty("minerva.hostVerification.enabled")
    .map { it.toBoolean() }
    .orElse(false)
val minervaRuntimeRoot = providers.gradleProperty("minerva.runtimeRoot")
val minervaCompilerScript = providers.gradleProperty("minerva.compilerScript")
val minervaKeyFile = providers.gradleProperty("minerva.keyFile")
val minervaCalibrationNpz = providers.gradleProperty("minerva.calibrationNpz")
val minervaRunCmakeBuild = providers.gradleProperty("minerva.hostVerification.runCmakeBuild")
val minervaRunCTest = providers.gradleProperty("minerva.hostVerification.runCTest")
val minervaHostOutputPath = providers.gradleProperty("minerva.hostVerification.hostOutputPath")
val minervaHostAdapterSource = providers.gradleProperty("minerva.hostVerification.hostAdapterSource")
val minervaHostIncludeDirs = providers.gradleProperty("minerva.hostVerification.hostIncludeDirs")
val minervaHostLibraryDirs = providers.gradleProperty("minerva.hostVerification.hostLibraryDirs")
val minervaHostLibraries = providers.gradleProperty("minerva.hostVerification.hostLibraries")
val minervaRunCmakeBuildForSample = minervaRunCmakeBuild
    .orElse(providers.environmentVariable("MINERVA_RUN_CMAKE"))
    .orElse(minervaHostVerificationEnabled.map { it.toString() })
val minervaRunCTestForSample = minervaRunCTest
    .orElse(providers.environmentVariable("MINERVA_RUN_CTEST"))
    .orElse(minervaHostVerificationEnabled.map { it.toString() })

val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")

tasks.register("minervaHostVerification") {
    group = "verification"
    description = "Runs external Minerva compiler and host verification when explicitly configured."
    enabled = minervaHostVerificationEnabled.get() &&
        minervaRuntimeRoot.isPresent &&
        minervaCompilerScript.isPresent
    if (enabled) {
        dependsOn("jvmTest", "runMinervaTinyMlpSample")
    }
    inputs.property("minerva.runtimeRoot", minervaRuntimeRoot.orElse(""))
    inputs.property("minerva.compilerScript", minervaCompilerScript.orElse(""))
    inputs.property("minerva.calibrationNpz", minervaCalibrationNpz.orElse(""))
    inputs.property("minerva.keyFile", minervaKeyFile.orElse(""))
    inputs.property("minerva.hostVerification.runCmakeBuild", minervaRunCmakeBuildForSample)
    inputs.property("minerva.hostVerification.runCTest", minervaRunCTestForSample)
    inputs.property("minerva.hostVerification.hostOutputPath", minervaHostOutputPath.orElse(""))
    inputs.property("minerva.hostVerification.hostAdapterSource", minervaHostAdapterSource.orElse(""))
    inputs.property("minerva.hostVerification.hostIncludeDirs", minervaHostIncludeDirs.orElse(""))
    inputs.property("minerva.hostVerification.hostLibraryDirs", minervaHostLibraryDirs.orElse(""))
    inputs.property("minerva.hostVerification.hostLibraries", minervaHostLibraries.orElse(""))
}

tasks.register<JavaExec>("runMinervaTinyMlpSample") {
    group = "application"
    description = "Runs the maintained Minerva tiny MLP export sample."

    dependsOn(tasks.named("jvmJar"))

    mainClass.set("sk.ainet.compile.minerva.examples.MinervaTinyMlpExportSample")
    workingDir = rootProject.projectDir

    classpath = files(
        jvmMainCompilation.runtimeDependencyFiles,
        tasks.named("jvmJar").get().outputs.files
    )

    minervaCompilerScript.orElse(providers.environmentVariable("MINERVA_COMPILER_SCRIPT")).orNull?.let {
        environment("MINERVA_COMPILER_SCRIPT", it)
    }
    minervaRuntimeRoot.orElse(providers.environmentVariable("MINERVA_RUNTIME_ROOT")).orNull?.let {
        environment("MINERVA_RUNTIME_ROOT", it)
    }
    minervaKeyFile.orElse(providers.environmentVariable("MINERVA_KEY_FILE")).orNull?.let {
        environment("MINERVA_KEY_FILE", it)
    }
    minervaCalibrationNpz.orElse(providers.environmentVariable("MINERVA_CALIBRATION_NPZ")).orNull?.let {
        environment("MINERVA_CALIBRATION_NPZ", it)
    }
    minervaRunCmakeBuildForSample.orNull?.let {
        environment("MINERVA_RUN_CMAKE", it)
    }
    minervaRunCTestForSample.orNull?.let {
        environment("MINERVA_RUN_CTEST", it)
    }
    minervaHostOutputPath.orElse(providers.environmentVariable("MINERVA_HOST_OUTPUT_PATH")).orNull?.let {
        environment("MINERVA_HOST_OUTPUT_PATH", it)
    }
    minervaHostAdapterSource.orElse(providers.environmentVariable("MINERVA_HOST_ADAPTER_SOURCE")).orNull?.let {
        environment("MINERVA_HOST_ADAPTER_SOURCE", it)
    }
    minervaHostIncludeDirs.orElse(providers.environmentVariable("MINERVA_HOST_INCLUDE_DIRS")).orNull?.let {
        environment("MINERVA_HOST_INCLUDE_DIRS", it)
    }
    minervaHostLibraryDirs.orElse(providers.environmentVariable("MINERVA_HOST_LIBRARY_DIRS")).orNull?.let {
        environment("MINERVA_HOST_LIBRARY_DIRS", it)
    }
    minervaHostLibraries.orElse(providers.environmentVariable("MINERVA_HOST_LIBRARIES")).orNull?.let {
        environment("MINERVA_HOST_LIBRARIES", it)
    }
}
