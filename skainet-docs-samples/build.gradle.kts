import org.jetbrains.kotlin.gradle.dsl.JvmTarget

/**
 * Documentation samples module.
 *
 * The Kotlin sources under this module ARE the Antora example resources: the
 * `commonMain` source directory points at `docs/modules/ROOT/examples/kotlin`, so
 * every snippet rendered in the docs is real code that this module compiles and
 * `commonTest` executes in CI (JVM-only samples: `examples/kotlin-jvm`, run by `jvmTest`). AsciiDoc pages pull tagged regions out of these files
 * with `include::example$kotlin/...[tag=...]` — there are no hand-typed snippets to rot.
 *
 * Not published: this module exists only to keep the documentation honest.
 */
plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }

    sourceSets {
        commonMain {
            // The example sources live in the docs tree so Antora can include them.
            kotlin.srcDir("../docs/modules/ROOT/examples/kotlin")
            dependencies {
                implementation(project(":skainet-lang:skainet-lang-core"))
                implementation(project(":skainet-backends:skainet-backend-cpu"))
                implementation(project(":skainet-compile:skainet-compile-dag"))
                implementation(project(":skainet-data:skainet-data-api"))
                implementation(project(":skainet-data:skainet-data-simple"))
            }
        }
        jvmMain {
            // Samples that need files (GGUF, memory mapping) are JVM-only and live beside the
            // common ones, so Antora includes them the same way: `example$kotlin-jvm/...`.
            kotlin.srcDir("../docs/modules/ROOT/examples/kotlin-jvm")
            dependencies {
                implementation(project(":skainet-backends:skainet-backend-api"))
                implementation(project(":skainet-io:skainet-io-core"))
                implementation(project(":skainet-io:skainet-io-gguf"))
                implementation(project(":skainet-frontend:skainet-frontend-core"))
                implementation(project(":skainet-frontend:skainet-frontend-text"))
            }
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
        }
    }
}
