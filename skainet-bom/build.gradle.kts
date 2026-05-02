plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

// Override the engine-wide `sk.ainet.core` group: downstream BOM
// consumers (e.g. sk.ainet.transformers:skainet-transformers-bom)
// import this with `<groupId>sk.ainet</groupId>`. vanniktech's
// auto-coordinates feature otherwise picks up GROUP=sk.ainet.core
// from the root gradle.properties and would publish the BOM at the
// wrong coordinates.
mavenPublishing {
    coordinates(
        groupId = "sk.ainet",
        artifactId = "skainet-bom",
        version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified"),
    )
}

group = "sk.ainet"
version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified")

javaPlatform {
    allowDependencies()
}

dependencies {
    constraints {
        // Core language module
        api(project(":skainet-lang:skainet-lang-core"))

        // Backend abstraction + CPU backend
        api(project(":skainet-backends:skainet-backend-api"))
        api(project(":skainet-backends:skainet-backend-cpu"))
        // Native (FFM) priority-100 kernel provider — bundles a
        // libskainet_kernels shared library and overrides the
        // priority-50 Panama kernels for Q4_K and FP32 matmul on
        // hosts where the native lib resolves. Cascades to Panama
        // otherwise (missing arch, sandbox, kill-switch).
        api(project(":skainet-backends:skainet-backend-native-cpu"))

        // IO modules
        api(project(":skainet-io:skainet-io-core"))
        api(project(":skainet-io:skainet-io-gguf"))
        api(project(":skainet-io:skainet-io-safetensors"))
        api(project(":skainet-io:skainet-io-onnx"))
        api(project(":skainet-io:skainet-io-image"))

        // Data modules
        api(project(":skainet-data:skainet-data-api"))
        api(project(":skainet-data:skainet-data-simple"))
        api(project(":skainet-data:skainet-data-transform"))

        // Compilation
        api(project(":skainet-compile:skainet-compile-core"))
        api(project(":skainet-compile:skainet-compile-hlo"))

        // Pipeline
        api(project(":skainet-pipeline"))

        // Models
        api(project(":skainet-models:skainet-model-yolo"))
    }
}
