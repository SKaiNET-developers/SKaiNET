plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
    id("sk.ainet.transformers.bom-coverage")
}

// Override the engine-wide `sk.ainet.core` group: downstream BOM
// consumers (e.g. sk.ainet.transformers:skainet-transformers-bom)
// import this with `<groupId>sk.ainet</groupId>`. vanniktech's
// auto-coordinates feature otherwise picks up GROUP=sk.ainet.core
// from the root gradle.properties and would publish the BOM at the
// wrong coordinates.
mavenPublishing {
    coordinates(
        groupId = "sk.ainet",
        artifactId = "skainet-bom",
        version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified"),
    )
}

group = "sk.ainet"
version = providers.gradleProperty("VERSION_NAME").getOrElse("unspecified")

javaPlatform {
    allowDependencies()
}

// Constraints are populated by sk.ainet.transformers.bom-coverage:
// every sibling subproject applying `com.vanniktech.maven.publish`
// is added as an `api` constraint, sorted by Gradle path. To skip
// a published module from the BOM, add its path to
// `bomCoverage.excludePublished`.
