plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.2.21"
    kotlin("plugin.serialization") version "2.3.10"
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
    implementation("org.jetbrains.dokka:dokka-gradle-plugin:2.1.0")
    implementation(gradleApi())
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(21)
}

gradlePlugin {
    plugins {
        register("SKaiNetDocumentation") {
            id = "sk.ainet.documentation"
            implementationClass = "DocumentationPlugin"
        }
    }
}
