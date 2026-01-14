plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    application
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)
    
    // SKaiNET dependencies
    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-lang:skainet-lang-models"))
    implementation(project(":skainet-compile:skainet-compile-dag"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))

    testImplementation(kotlin("test"))
}

application {
    mainClass.set("sk.ainet.apps.sine.SineApproxCliKt")
}
