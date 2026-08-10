plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties (incl. the
// wasmJs executable flag); explicitApi() and kotlin-test come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.compilie.dag"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(project(":skainet-lang:skainet-lang-core"))
            api(project(":skainet-lang:skainet-lang-dag"))
            api(project(":skainet-compile:skainet-compile-core"))
        }

        commonTest.dependencies {
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-lang:skainet-lang-models"))
        }
    }
}
