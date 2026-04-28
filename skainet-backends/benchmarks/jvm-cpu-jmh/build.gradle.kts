plugins {
    kotlin("jvm")
    id("me.champeau.jmh") version "0.7.3"
}

dependencies {
    implementation(project(":skainet-lang:skainet-lang-core"))
    implementation(project(":skainet-backends:skainet-backend-api"))
    implementation(project(":skainet-backends:skainet-backend-cpu"))
}

jmh {
    fork.set(1)
    warmupIterations.set(3)
    iterations.set(5)
    //timeOnIteration.set(org.gradle.api.tasks.testing.logging.TestLogEvent.values().size.toLong())
    jvmArgs.set(listOf("--enable-preview", "--add-modules", "jdk.incubator.vector"))
}

// Ensure JMH also gets the incubator module args when running from IDE Gradle
tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}
