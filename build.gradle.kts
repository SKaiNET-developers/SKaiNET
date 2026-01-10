plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.asciidoctorJvm) apply false
    alias(libs.plugins.dokka) apply false
    alias(libs.plugins.skainet.docs)
    id("org.jetbrains.kotlinx.benchmark") version "0.4.13" apply false
}

allprojects {
    group = "sk.ainet"
}

// Enforce a consistent JVM toolchain across all Kotlin modules to avoid bytecode mismatches
subprojects {
    // Kotlin Multiplatform projects
    plugins.withId("org.jetbrains.kotlin.multiplatform") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension::class.java)?.apply {
            jvmToolchain(21)
        }
    }
    // Kotlin/JVM projects
    plugins.withId("org.jetbrains.kotlin.jvm") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension::class.java)?.apply {
            jvmToolchain(21)
        }
    }

    tasks.withType<Test>().configureEach {
        maxHeapSize = "8192m"
    }
}

kover {
    reports {
        total {
            html {
                onCheck = true
            }
            xml {
                onCheck = true
            }
        }
    }
}

// Custom task to generate operator documentation
tasks.register("generateOperatorDocs") {
    group = "documentation"
    description = "Generate operator documentation from KSP-generated JSON files"
    
    // Configure inputs for incremental builds
    inputs.files("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json")
    // Configure outputs for incremental builds
    outputs.dir("docs/modules/operators/_generated_")
    outputs.cacheIf { true }
    
    // Depend on KSP processing
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
    
    // Run built-in documentation generation task (provided by sk.ainet.documentation plugin)
    dependsOn("generateDocs")
    
    doLast {
        println("Operator documentation generation completed")
    }
}

// Documentation plugin configuration
documentation {
    inputFile.set(file("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json"))
    outputDirectory.set(file("docs/modules/operators/_generated_"))
    includeBackendStatus.set(true)
    generateIndex.set(true)
}

tasks.named("generateDocs") {
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
}