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
            // `api`: kotlinx.io.Source / Sink are part of this module's public API (DataSourceArtifact.openSource(),
            // copyTo(), DataSourceRemoteContent, DataSourceArtifactStore), so consumers need them on their compile
            // classpath. As `implementation` a consumer could not call those functions without adding kotlinx-io itself.
            api(libs.kotlinx.io.core)
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
