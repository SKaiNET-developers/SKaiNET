plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "sk.ainet"
version = rootProject.findProperty("VERSION_NAME") ?: "0.14.0"

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
