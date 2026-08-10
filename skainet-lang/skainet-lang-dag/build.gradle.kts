plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.ksp)
    id("sk.ainet.dokka")
}

// Targets come from skainet.targets in this module's gradle.properties;
// explicitApi() and kotlin-test in commonTest come from sk.ainet.multiplatform.
skainet {
    namespace = "sk.ainet.lang.dag"
}

kotlin {
    sourceSets {
        commonMain {
            kotlin.srcDir("build/generated/ksp/metadata/commonMain/kotlin")
            dependencies {
                api(project(":skainet-lang:skainet-lang-core"))
                implementation(libs.kotlinx.coroutines)
            }
        }

        commonTest.dependencies {
            implementation(project(":skainet-backends:skainet-backend-cpu"))
        }
    }
}

// Ensure KSP metadata task runs before any Kotlin compilation, other KSP tasks, and sourcesJar
tasks.configureEach {
    if (name != "kspCommonMainKotlinMetadata" &&
        (name.startsWith("compileKotlin") || name.startsWith("ksp") || name.contains("ourcesJar"))
    ) {
        dependsOn("kspCommonMainKotlinMetadata")
    }
}

tasks.matching { it.name.startsWith("dokka") }.configureEach {
    dependsOn("kspCommonMainKotlinMetadata")
}

dependencies {
    add("kspCommonMainMetadata", project(":skainet-lang:skainet-lang-ksp-processor"))
}
