import org.jetbrains.dokka.gradle.DokkaExtension
import org.jetbrains.dokka.gradle.engine.parameters.VisibilityModifier

plugins {
    id("org.jetbrains.dokka")
}

extensions.configure<DokkaExtension> {
    moduleName.set(project.name)
    moduleVersion.set(providers.gradleProperty("VERSION_NAME"))

    dokkaPublications.configureEach {
        suppressInheritedMembers.set(true)
    }

    dokkaSourceSets.configureEach {
        documentedVisibilities(VisibilityModifier.Public)
        suppressGeneratedFiles.set(true)

        // Suppress native source sets that use cinterop — Dokka 2.x cannot translate
        // platform-specific interop symbols (e.g. CoreGraphics, posix).
        val nativeSuffixes = listOf(
            "iosArm64Main", "iosSimulatorArm64Main",
            "macosArm64Main",
            "linuxX64Main", "linuxArm64Main",
        )
        if (nativeSuffixes.any { name.endsWith(it) }) {
            suppress.set(true)
        }

        sourceLink {
            localDirectory.set(projectDir.resolve("src"))
            remoteUrl("https://github.com/SKaiNET-developers/skainet/tree/main/${project.path.replace(":", "/").removePrefix("/")}/src")
            remoteLineSuffix.set("#L")
        }
    }
}
