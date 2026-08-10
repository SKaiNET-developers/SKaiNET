plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties;
// explicitApi() and kotlin-test in commonTest come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.core.api"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(libs.kotlinx.coroutines)
        }

        commonTest.dependencies {
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
