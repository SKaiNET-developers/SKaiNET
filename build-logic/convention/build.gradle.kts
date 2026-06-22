plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.3.21"
    kotlin("plugin.serialization") version "2.4.0"
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
    implementation("org.jetbrains.dokka:dokka-gradle-plugin:2.2.0")
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
        register("SKaiNetBomCoverage") {
            id = "sk.ainet.transformers.bom-coverage"
            implementationClass = "sk.ainet.buildlogic.bom.BomCoveragePlugin"
        }
    }
}
