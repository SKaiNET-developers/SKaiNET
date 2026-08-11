plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)

    id("sk.ainet.dokka")
}

// The default SKaiNET target set — jvm, js, wasmJs, wasmWasi, apple, linux (plus the
// Android target from the AGP plugin above) — together with explicitApi() and
// kotlin-test in commonTest, all come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.pipeline"
}

// No dependencies on skainet-lang-core - keep it minimal
