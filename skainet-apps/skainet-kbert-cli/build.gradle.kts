plugins {
    kotlin("jvm")
    alias(libs.plugins.shadow)
}

dependencies {
    implementation(project(":skainet-apps:skainet-bert"))
    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-io:skainet-io-core"))
    implementation(project(":skainet-io:skainet-io-safetensors"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))
    implementation(libs.kotlinx.coroutines)
}

tasks.withType<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar> {
    archiveBaseName.set("kbert")
    archiveClassifier.set("all")
    archiveVersion.set("")

    manifest {
        attributes(
            "Main-Class" to "sk.ainet.apps.bert.cli.MainKt",
            "Add-Opens" to "java.base/jdk.internal.misc",
            "Multi-Release" to "true"
        )
    }

    mergeServiceFiles()
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
