plugins {
    alias(libs.plugins.jetbrainsKotlinJvm)
    application
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(libs.kotlinx.cli)

    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-io:skainet-io-core"))
    implementation(project(":skainet-io:skainet-io-gguf"))

    testImplementation(kotlin("test"))
    testImplementation(project(":skainet-io:skainet-io-gguf"))
}

application {
    mainClass.set("sk.ainet.apps.plan.SkainetPlanKt")
    // the table uses · ✔ ✘ ← — print them as UTF-8 regardless of the terminal locale
    applicationDefaultJvmArgs = listOf("-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8", "-Dfile.encoding=UTF-8")
}

tasks.test {
    useJUnitPlatform()
}
