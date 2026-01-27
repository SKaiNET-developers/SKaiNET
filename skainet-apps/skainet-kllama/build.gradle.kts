import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    id("com.android.library") apply false
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.shadow) apply false
}

if (!project.hasProperty("buildFatJar")) {
    plugins.apply("com.android.library")
}

// Separate configuration for Shadow to avoid conflict with Android plugin when applying both
// Shadow is only needed for the JVM target
if (project.hasProperty("buildFatJar")) {
    plugins.apply(libs.plugins.shadow.get().pluginId)
}

kotlin {
    jvmToolchain(21)
    
    // Disable Android target when building fat JAR to avoid plugin conflicts
    if (!project.hasProperty("buildFatJar")) {
        androidTarget {
            publishLibraryVariants("release")
            @OptIn(ExperimentalKotlinGradlePluginApi::class)
            compilerOptions {
                jvmTarget.set(JvmTarget.JVM_11)
            }
        }
    }

    iosArm64()
    iosSimulatorArm64()

    macosArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    linuxX64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    linuxArm64 {
        binaries {
            executable {
                entryPoint = "sk.ainet.apps.kllama.cli.main"
                baseName = "kllama"
            }
        }
    }

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-backends:skainet-backend-cpu"))
            implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
            implementation(project(":skainet-io:skainet-io-core"))
            implementation(project(":skainet-io:skainet-io-gguf"))
            implementation(libs.kotlinx.io.core)
            implementation(libs.kotlinx.coroutines)

        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-lang:skainet-lang-models"))
            implementation(project(":skainet-io:skainet-io-gguf"))
        }

        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
                implementation(libs.kotlinx.coroutines.test)
                implementation(project(":skainet-backends:skainet-backend-cpu"))
            }
        }
        // val androidMain by getting
        if (!project.hasProperty("buildFatJar")) {
            val androidMain by getting
        }
        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }

        val nativeMain by creating {
            dependsOn(commonMain.get())
        }
        val linuxX64Main by getting { dependsOn(nativeMain) }
        val linuxArm64Main by getting { dependsOn(nativeMain) }
        val macosArm64Main by getting { dependsOn(nativeMain) }
        val iosArm64Main by getting { dependsOn(nativeMain) }
        val iosSimulatorArm64Main by getting { dependsOn(nativeMain) }
    }
}


tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

if (project.hasProperty("buildFatJar")) {
    val shadowJar = tasks.named<Jar>("shadowJar") {
        archiveBaseName.set("kllama")
        archiveClassifier.set("fat")
        manifest {
            attributes["Main-Class"] = "sk.ainet.apps.kllama.cli.MainKt"
        }
    }
}

if (!project.hasProperty("buildFatJar")) {
    configure<com.android.build.gradle.LibraryExtension> {
        namespace = "sk.ainet.apps.kllama"
        compileSdk = libs.versions.android.compileSdk.get().toInt()

        defaultConfig {
            minSdk = libs.versions.android.minSdk.get().toInt()
        }
        compileOptions {
            sourceCompatibility = JavaVersion.VERSION_11
            targetCompatibility = JavaVersion.VERSION_11
        }
    }
}

