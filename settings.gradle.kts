pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}




rootProject.name = "SKaiNET"

includeBuild("build-logic")

// ====== LANG
include("skainet-lang:skainet-lang-core")
include("skainet-lang:skainet-lang-audio")
include("skainet-lang:skainet-lang-models")
include("skainet-lang:skainet-lang-ksp-annotations")
include("skainet-lang:skainet-lang-ksp-processor")
include("skainet-lang:skainet-kan")
include("skainet-lang:skainet-lang-dag")


// ====== COMPILE
include("skainet-compile:skainet-compile-core")
include("skainet-compile:skainet-compile-dag")
include("skainet-compile:skainet-compile-json")
include("skainet-compile:skainet-compile-hlo")
include("skainet-compile:skainet-compile-c")

// ====== BACKENDS
include("skainet-backends:skainet-backend-cpu")

// ====== BENCHMARKS
include("skainet-backends:benchmarks:jvm-cpu-jmh")

// ====== DATA
include("skainet-data:skainet-data-api")
include("skainet-data:skainet-data-transform")
include("skainet-data:skainet-data-simple")
include("skainet-data:skainet-data-media")

// ====== PIPELINE
include("skainet-pipeline")

// ====== IO
include("skainet-io:skainet-io-core")
include("skainet-io:skainet-io-gguf")
include("skainet-io:skainet-io-image")
include("skainet-io:skainet-io-onnx")
include("skainet-io:skainet-io-wav")

// ====== models
include("skainet-models:skainet-model-yolo")

// ====== Integrations
//include("skainet-integrations:skainet-simple-cpu")


// ====== TEST
include("skainet-test:skainet-test-groundtruth")

// ====== APPS
include("skainet-apps:skainet-grayscale-cli")
include("skainet-apps:skainet-tensor-tools")
include("skainet-apps:skainet-llm")
include("skainet-apps:skainet-bert")
include("skainet-apps:skainet-kllama")
include("skainet-apps:skainet-kllama-cli")
include("skainet-apps:skainet-kgemma")
include("skainet-apps:skainet-kbert-cli")
include("skainet-io:skainet-io-safetensors")
