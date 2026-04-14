import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}


kotlin {
    jvm()
    explicitApi()

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()

    // Android Native targets for vendor-specific backends linking directly against
    // libneuralnetworks.so / libOpenCL.so / etc. (e.g. skainet-backend-nnapi).
    androidNativeArm32()
    androidNativeArm64()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }
}

