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
    // Run a subset: ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh -PjmhIncludes='spike.*'
    (project.findProperty("jmhIncludes") as String?)?.let { includes.set(listOf(it)) }
    //timeOnIteration.set(org.gradle.api.tasks.testing.logging.TestLogEvent.values().size.toLong())
    jvmArgs.set(listOf("--enable-preview", "--add-modules", "jdk.incubator.vector"))
}

// Ensure JMH also gets the incubator module args when running from IDE Gradle
tasks.withType<JavaExec>().configureEach {
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector")
}

// SKEEP-003 Phase-2 spike (#1016): flat-RSS decode loop, Forward slab vs heap arrays.
// ./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:runFlatRssSpike -Pspike.steps=2000 -Pspike.mode=both
tasks.register<JavaExec>("runFlatRssSpike") {
    group = "benchmark"
    description = "SKEEP-003 P2 spike: RSS over decode steps with a recycled Forward slab vs heap arrays"
    classpath = sourceSets["jmh"].runtimeClasspath
    mainClass.set("sk.ainet.bench.spike.FlatRssSpike")
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-Xmx256m")
    args((project.findProperty("spike.steps") as String?) ?: "2000", (project.findProperty("spike.mode") as String?) ?: "both")
}
