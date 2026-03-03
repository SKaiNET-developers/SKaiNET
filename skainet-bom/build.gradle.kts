plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "sk.ainet"
version = rootProject.findProperty("VERSION_NAME") ?: "0.13.0"

javaPlatform {
    allowDependencies()
}

dependencies {
    constraints {
        // Core language module
        api("sk.ainet:skainet-lang-core-jvm:$version")

        // CPU backend
        api("sk.ainet:skainet-backend-cpu-jvm:$version")

        // IO modules
        api("sk.ainet:skainet-io-core-jvm:$version")
        api("sk.ainet:skainet-io-gguf-jvm:$version")
        api("sk.ainet:skainet-io-safetensors-jvm:$version")
        api("sk.ainet:skainet-io-onnx-jvm:$version")
        api("sk.ainet:skainet-io-image-jvm:$version")

        // Data modules
        api("sk.ainet:skainet-data-api-jvm:$version")
        api("sk.ainet:skainet-data-simple-jvm:$version")
        api("sk.ainet:skainet-data-transform-jvm:$version")

        // LLM inference
        api("sk.ainet:skainet-kllama-jvm:$version")
        api("sk.ainet:skainet-kllama-agent-jvm:$version")

        // BERT
        api("sk.ainet:skainet-bert-jvm:$version")

        // LLM common
        api("sk.ainet:skainet-llm-jvm:$version")

        // Compilation
        api("sk.ainet:skainet-compile-core-jvm:$version")
        api("sk.ainet:skainet-compile-hlo-jvm:$version")
    }
}
