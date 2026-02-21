plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.3.10"
    kotlin("plugin.serialization") version "2.2.21"
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}

group = "sk.ainet.buildlogic"

dependencies {
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.optimumcode.json.schema.validator)
    implementation(libs.asciidoctorj.core)
    implementation(gradleApi())
}

kotlin {
    jvmToolchain(21)
}

gradlePlugin {
    plugins {
        register("SKaiNetDocumentation") {
            id = "sk.ainet.documentation"
            implementationClass = "DocumentationPlugin"
        }
    }
}
