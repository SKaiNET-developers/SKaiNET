plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties. The previous
// hand-wired native source-set tree carried no source files and is gone — the default
// hierarchy template covers this module.
skainet {
    namespace = "sk.ainet.backend.api"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            // Neutral backend API is an `api` re-export of the tensor op and
            // storage interfaces already defined in skainet-lang-core. Any
            // concrete backend (CPU, IREE, Metal, NPU, ...) should depend on
            // this module instead of pulling in skainet-backend-cpu just to
            // reach TensorOps / TensorDataFactory / TensorData.
            api(project(":skainet-lang:skainet-lang-core"))
        }
    }
}
