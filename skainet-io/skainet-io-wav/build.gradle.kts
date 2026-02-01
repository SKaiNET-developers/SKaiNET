import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
}

kotlin {
    jvm()
    macosArm64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":skainet-lang:skainet-lang-core"))
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
        val appleMain by creating {
            dependsOn(commonMain)
        }
        val jvmMain by getting
        val jvmTest by getting
        val macosArm64Main by getting {
            dependsOn(appleMain)
        }
        val iosArm64Main by getting {
            dependsOn(appleMain)
        }
        val iosSimulatorArm64Main by getting {
            dependsOn(appleMain)
        }
    }
}
