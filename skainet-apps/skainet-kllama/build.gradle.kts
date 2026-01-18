import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.shadow)
}

kotlin {
    jvmToolchain(21)

    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
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
        val androidMain by getting
        val wasmJsMain by getting
        wasmJsMain.dependencies {
            implementation(libs.kotlinx.browser)
        }

        val commonMain by getting

        val nativeMain by creating {
            dependsOn(commonMain)
        }

        val linuxMain by creating {
            dependsOn(nativeMain)
        }

        val iosMain by creating {
            dependsOn(nativeMain)
        }

        val macosMain by creating {
            dependsOn(nativeMain)
        }

        val iosArm64Main by getting {
            dependsOn(iosMain)
        }

        val iosSimulatorArm64Main by getting {
            dependsOn(iosMain)
        }

        val macosArm64Main by getting {
            dependsOn(macosMain)
        }

        val linuxX64Main by getting {
            dependsOn(linuxMain)
        }

        val linuxArm64Main by getting {
            dependsOn(linuxMain)
        }
    }
}


tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

android {
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

// Shadow JAR configuration for JVM fat JAR
tasks.register<ShadowJar>("shadowJar") {
    archiveBaseName.set("kllama")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.kllama.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true"
        )
    }

    from(kotlin.jvm().compilations.getByName("main").output)
    configurations = listOf(project.configurations.getByName("jvmRuntimeClasspath"))

    // Merge service files for proper SPI support
    mergeServiceFiles()

    dependsOn("jvmJar")
}
