import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
}

kotlin {
    explicitApi()
    android {
        namespace = "sk.ainet.backend.cpu"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmWasi {
        nodejs()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))

        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(project(":skainet-lang:skainet-lang-models"))
        }

        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
        val androidMain by getting
        val wasmJsMain by getting

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
