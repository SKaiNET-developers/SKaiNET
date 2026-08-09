plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties (androidNative for
// vendor-specific backends linking against libneuralnetworks.so / libOpenCL.so / etc.).
