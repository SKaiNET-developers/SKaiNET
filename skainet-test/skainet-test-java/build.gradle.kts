plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.0")
    testImplementation(project(":skainet-lang:skainet-lang-core"))
    testImplementation(project(":skainet-backends:skainet-backend-cpu"))
    testImplementation(project(":skainet-data:skainet-data-simple"))
}

tasks.test {
    useJUnitPlatform()
    jvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector")
    maxHeapSize = "4g"
}
