plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties;
// explicitApi() and kotlin-test in commonTest come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.compilie.core"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
        }
    }
}
