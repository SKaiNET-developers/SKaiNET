plugins {
    // AGP 9 ships built-in Kotlin support — applying kotlin("android") is an
    // error since 9.0, so this module uses the android plugin alone.
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

/*
 * Android JNI bridge for the hand-written C/NEON matmul kernels
 * (skainet-backend-native-cpu/native). ART has no java.lang.foreign, so the
 * FFM provider can never run on Android — this module ships the same C
 * sources as an AAR (.so via externalNativeBuild/NDK) with thin JNI shims
 * and a priority-100 KernelProvider discovered via ServiceLoader (#920).
 *
 * Two shared libs are built from the same sources (see native/CMakeLists.txt):
 *   libskainet_jni.so      — baseline armv8-a (NEON always present on arm64;
 *                            runs on every device incl. Cortex-A53)
 *   libskainet_jni_v82.so  — -march=armv8.2-a+fp16+dotprod (vdotq_s32 paths
 *                            in q4k/q6k; would SIGILL on armv8.0 cores)
 * JniKernels picks ONE at load time from /proc/cpuinfo features — runtime
 * dispatch without symbol renaming or ifunc.
 */
android {
    namespace = "sk.ainet.exec.kernel.jni"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    // Pinned NDK so the AAR's .so's build reproducibly locally and on the
    // release runner (AGP provisions it via sdkmanager when absent).
    ndkVersion = libs.versions.android.ndk.get()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            // arm64 is the target that matters; x86_64 keeps emulator CI and
            // desktop AVDs working (scalar C paths). 32-bit ARM is out of
            // scope — different intrinsics story, shrinking device share.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=none"
            }
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    externalNativeBuild {
        cmake {
            path = file("native/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
        }
    }

    // Publishing variant selection is configured by the vanniktech
    // maven-publish plugin (AndroidSingleVariantLibrary "release").
}

dependencies {
    api(project(":skainet-backends:skainet-backend-api"))

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test)

    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
    // Scalar reference kernels for on-device parity checks.
    androidTestImplementation(project(":skainet-backends:skainet-backend-cpu"))
    // The #1130 (M2-A5) measurement harness loads a real GGUF on-device.
    androidTestImplementation(project(":skainet-io:skainet-io-core"))
    androidTestImplementation(project(":skainet-io:skainet-io-gguf"))
    androidTestImplementation(libs.kotlinx.coroutines)
}
