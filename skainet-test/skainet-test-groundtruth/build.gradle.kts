import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
}

// =============================================================================
// Ground Truth Generation via Docker
// =============================================================================

// Path to the skainet-ground-truth project (relative to SKaiNET root)
val groundTruthProjectDir = rootProject.projectDir.parentFile.resolve("skainet-ground-truth/pytorch")
val groundTruthResultsDir = groundTruthProjectDir.resolve("results")

// Docker image name for ground truth generation
val groundTruthDockerImage = "skainet-ground-truth"

/**
 * Build the Docker image for ground truth generation.
 * Only needs to be run once or when Dockerfile changes.
 */
tasks.register<Exec>("buildGroundTruthDocker") {
    group = "ground truth"
    description = "Build the Docker image for PyTorch ground truth generation"

    workingDir = groundTruthProjectDir
    commandLine("docker", "build", "-t", groundTruthDockerImage, ".")

    doFirst {
        println("Building Docker image '$groundTruthDockerImage' from ${groundTruthProjectDir}")
    }
}

/**
 * Generate ground truth GGUF files by running the Docker container.
 * This executes PyTorch operations and stores results in GGUF format.
 */
tasks.register<Exec>("generateGroundTruth") {
    group = "ground truth"
    description = "Generate GGUF ground truth files using PyTorch in Docker"

    workingDir = groundTruthProjectDir

    commandLine(
        "docker", "run", "--rm",
        "-v", "${groundTruthProjectDir.resolve("src")}:/app/src",
        "-v", "${groundTruthResultsDir}:/app/results",
        groundTruthDockerImage
    )

    doFirst {
        println("Generating ground truth GGUF files...")
        println("  Source: ${groundTruthProjectDir.resolve("src")}")
        println("  Output: $groundTruthResultsDir")
    }

    doLast {
        println("Ground truth generation complete.")
        println("GGUF files available at: $groundTruthResultsDir")
    }
}

/**
 * Clean generated ground truth files.
 */
tasks.register<Delete>("cleanGroundTruth") {
    group = "ground truth"
    description = "Delete generated ground truth GGUF files"

    delete(groundTruthResultsDir)

    doLast {
        println("Cleaned ground truth results directory")
    }
}

/**
 * List available ground truth test suites.
 */
tasks.register("listGroundTruth") {
    group = "ground truth"
    description = "List available ground truth test suites and GGUF files"

    doLast {
        if (groundTruthResultsDir.exists()) {
            println("Ground truth GGUF files in: $groundTruthResultsDir")
            groundTruthResultsDir.walkTopDown()
                .filter { it.isFile && it.extension == "gguf" }
                .forEach { println("  - ${it.relativeTo(groundTruthResultsDir)}") }
        } else {
            println("No ground truth files found. Run './gradlew generateGroundTruth' first.")
        }
    }
}

// Make ground truth results available as a system property for tests
tasks.withType<Test> {
    systemProperty("groundtruth.results.dir", groundTruthResultsDir.absolutePath)
}

kotlin {
    explicitApi()

    android {
        namespace = "sk.ainet.test.groundtruth"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    jvm()

    sourceSets {
        commonMain.dependencies {
            // Core tensor and operations
            api(project(":skainet-lang:skainet-lang-core"))

            // GGUF file reading
            api(project(":skainet-io:skainet-io-gguf"))

            // CPU backend for executing operations
            api(project(":skainet-backends:skainet-backend-cpu"))

            // IO utilities
            implementation(libs.kotlinx.io.core)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        jvmMain.dependencies {
            // JVM-specific dependencies if needed
        }

        jvmTest.dependencies {
            implementation(libs.junit)
        }
    }
}
