plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    application
    alias(libs.plugins.shadow)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    implementation(libs.kotlinx.serialization.json)

    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-backends:skainet-backend-api"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))
    // The ternary scenario benches the vendored NeoGPU LUT kernel through its
    // FFM face (#1141) — the native module bundles libskainet_kernels as a
    // jar resource, so the benchmark runs it exactly as a consumer would.
    implementation(project(":skainet-backends:skainet-backend-native-cpu"))

    testImplementation(kotlin("test"))
    testImplementation(libs.kotest.runner.junit5)
    testImplementation(libs.kotest.assertions.core)
}

application {
    mainClass.set("sk.ainet.bench.publish.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.named<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJar") {
    archiveBaseName.set("skainet-engine-publish")
    archiveClassifier.set("all")
    mergeServiceFiles()
}
