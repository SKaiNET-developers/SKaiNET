plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// JVM-only; see skainet.targets in this module's gradle.properties.

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.serialization.json)
        }

        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }

        jvmMain.dependencies {
            implementation(libs.ktor.client.cio)
            implementation(libs.ktor.client.core)
            implementation(libs.ktor.client.plugins)
            implementation(libs.kotlinx.coroutines.core.jvm)
        }
    }
}
