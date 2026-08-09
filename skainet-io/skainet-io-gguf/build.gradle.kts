import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties (mingw per #911:
// its createRandomAccessSource actual routes to io-core's WindowsRandomAccessSource).
// explicitApi(), kotlin-test and -Xexpect-actual-classes come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.io.gguf"
    androidJvmTarget = JvmTarget.JVM_1_8
    expectActualClasses = true
    // Pre-migration behavior: this module never enabled explicit API mode; turning it on
    // is a separate cleanup from the #911 target work.
    explicitApi = false
}

kotlin {
    sourceSets {
        // This module opts out of the default hierarchy template
        // (kotlin.mpp.applyDefaultHierarchyTemplate=false in gradle.properties) — custom
        // dependsOn edges would silently disable it anyway — and wires the native tree by
        // hand: the posix `pread`-backed createRandomAccessSource actual is shared by the
        // Apple and Linux targets via `posixMain`, while mingwX64 hangs off `nativeMain`
        // directly with its own Win32-backed leaf actual (posix pread does not exist
        // there). No apple/linux intermediates: no sources live at that level.
        nativeMain { dependsOn(commonMain.get()) }
        val posixMain by creating { dependsOn(nativeMain.get()) }
        listOf(iosArm64Main, iosSimulatorArm64Main, macosArm64Main, linuxX64Main, linuxArm64Main)
            .forEach { it.get().dependsOn(posixMain) }
        mingwX64Main { dependsOn(nativeMain.get()) }

        commonMain.dependencies {
            implementation(libs.kotlinx.io.core)
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-io:skainet-io-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-compile:skainet-compile-dag"))
        }
        jvmTest.dependencies {
            implementation(libs.junit)
            implementation(libs.kotlinx.coroutines)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
