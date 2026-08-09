import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties. mingw is safe
// here without extra source: io-safetensors has no posix in its own nativeMain —
// createRandomAccessSource / readTextFile are stubs and currentTimeMillis uses a monotonic
// TimeSource — so there is no bit-width metadata issue (unlike io-core, whose posix pread
// needed the native64Main split). File-backed reads route through io-core's RandomAccessSource.
skainet {
    namespace = "sk.ainet.io.safetensors"
    androidJvmTarget = JvmTarget.JVM_1_8
    expectActualClasses = true
    // Pre-migration behavior: this module never enabled explicit API mode; turning it on
    // is a separate cleanup from the #911 target work.
    explicitApi = false
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-io:skainet-io-core"))
        }
        jvmTest.dependencies {
            implementation(libs.junit)
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.coroutines.test)
            implementation(project(":skainet-backends:skainet-backend-cpu"))
        }
    }
}
