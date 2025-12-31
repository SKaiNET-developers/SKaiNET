plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.kotlinSerialization)
    application
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines)
    
    // SKaiNET dependencies
    implementation(project(":skainet-lang:skainet-lang-models"))
    implementation(project(":skainet-io:skainet-io-image"))
    implementation(project(":skainet-compile:skainet-compile-hlo"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))

    testImplementation(kotlin("test"))
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.kotest.runner.junit5)
    testImplementation(libs.kotest.assertions.core)
    testImplementation(libs.kotest.property)
}

application {
    mainClass.set("sk.ainet.apps.grayscale.GrayscaleImageCliKt")
}

tasks.test {
    useJUnitPlatform()
}