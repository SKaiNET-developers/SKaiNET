plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    application
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.pbandk.runtime)

    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-lang:skainet-lang-models"))
    implementation(project(":skainet-io:skainet-io-core"))
    implementation(project(":skainet-io:skainet-io-onnx"))
    implementation(project(":skainet-io:skainet-io-image"))
    implementation(project(":skainet-io:skainet-io-gguf"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))
    implementation(project(":skainet-models:skainet-model-yolo"))
    implementation(libs.pbandk.runtime)

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("sk.ainet.apps.onnx.tools.WeightExportKt")
    applicationDefaultJvmArgs = listOf(
        "--add-modules=jdk.incubator.vector"
    )
}

tasks.test {
    useJUnitPlatform()
}
