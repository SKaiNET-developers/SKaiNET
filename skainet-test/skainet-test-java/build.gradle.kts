plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
    testImplementation(project(":skainet-lang:skainet-lang-core"))
    testImplementation(project(":skainet-backends:skainet-backend-cpu"))
    testImplementation(project(":skainet-data:skainet-data-simple"))
    // 0.19.0 Java consumption surface: converter factory, tokenizer
    // factory, and the TensorSpec encoding helper facade. Tested in
    // ReleaseApiJavaTest so a Java consumer of the upcoming release
    // has a reference invocation pattern for each.
    testImplementation(project(":skainet-compile:skainet-compile-hlo"))
    testImplementation(project(":skainet-io:skainet-io-core"))
}

tasks.test {
    useJUnitPlatform()
    jvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "4g"
}
