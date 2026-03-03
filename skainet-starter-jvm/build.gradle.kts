plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "sk.ainet"
version = rootProject.findProperty("VERSION_NAME") ?: "0.14.0"

kotlin {
    jvmToolchain(21)
}

dependencies {
    api(project(":skainet-lang:skainet-lang-core"))
    api(project(":skainet-backends:skainet-backend-cpu"))
    api(project(":skainet-data:skainet-data-simple"))
}
