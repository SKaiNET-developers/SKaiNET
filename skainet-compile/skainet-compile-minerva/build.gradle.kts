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
