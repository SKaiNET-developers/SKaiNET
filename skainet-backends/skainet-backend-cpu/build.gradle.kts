plugins {
    id("sk.ainet.multiplatform")
    alias(libs.plugins.androidMultiplatformLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.binary.compatibility.validator)
    id("sk.ainet.dokka")
}

skainet {
    namespace = "sk.ainet.backend.cpu"
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            // Every concrete backend should go through the neutral api
            // module; it transitively brings in skainet-lang-core.
            implementation(project(":skainet-backends:skainet-backend-api"))
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(project(":skainet-compile:skainet-compile-core"))
            implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
        }

        commonTest.dependencies {
            implementation(project(":skainet-lang:skainet-lang-models"))
        }

        jvmMain.dependencies {
            implementation(libs.kotlinx.coroutines)
        }

        // This module opts out of the default hierarchy template
        // (kotlin.mpp.applyDefaultHierarchyTemplate=false in gradle.properties),
        // so the native tree is wired by hand.
        nativeMain { dependsOn(commonMain.get()) }
        appleMain { dependsOn(nativeMain.get()) }
        linuxMain { dependsOn(nativeMain.get()) }
        iosMain { dependsOn(appleMain.get()) }
        macosMain { dependsOn(appleMain.get()) }

        iosArm64Main { dependsOn(iosMain.get()) }
        iosSimulatorArm64Main { dependsOn(iosMain.get()) }
        macosArm64Main { dependsOn(macosMain.get()) }
        linuxX64Main { dependsOn(linuxMain.get()) }
        linuxArm64Main { dependsOn(linuxMain.get()) }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
