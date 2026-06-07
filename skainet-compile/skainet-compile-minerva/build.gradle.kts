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

tasks.register("minervaHostVerification") {
    group = "verification"
    description = "Gated lifecycle hook for external Minerva host verification in CI."
    enabled = minervaHostVerificationEnabled.get() &&
        minervaRuntimeRoot.isPresent &&
        minervaCompilerScript.isPresent
    if (enabled) {
        dependsOn("jvmTest")
    }
    inputs.property("minerva.runtimeRoot", minervaRuntimeRoot.orElse(""))
    inputs.property("minerva.compilerScript", minervaCompilerScript.orElse(""))
}
