plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.ksp)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    jvm()
    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(libs.kotlinx.serialization.json)
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(libs.kotlinpoet) // Use version from libs.versions.toml
                implementation(libs.kotlinpoet.ksp) // Required for KSP integration
                implementation(libs.ksp.api)
                implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
            }
            kotlin.srcDir("src/main/kotlin")
            resources.srcDir("src/main/resources")
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-junit"))
                implementation(libs.kotlin.compile.testing)
                implementation(libs.kotlin.compile.testing.ksp)
                implementation(project(":skainet-lang:skainet-lang-core"))
            }
            kotlin.srcDir("src/test/kotlin")
            resources.srcDir("src/test/resources")
        }
    }
}
