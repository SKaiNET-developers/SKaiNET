package sk.ainet.buildlogic.npm

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.work.DisableCachingByDefault

/**
 * Fails when a committed Yarn lockfile disagrees with a declared npm pin.
 *
 * Yarn `resolutions` make the pin take effect, but only the next time the lockfile
 * is regenerated. Without this check a stale or hand-edited lockfile silently wins
 * — which is exactly how PR #894's `ws` bump ended up as a zero-line diff.
 */
@DisableCachingByDefault(because = "Reads two small lockfiles; caching costs more than it saves")
abstract class VerifyNpmPinsTask : DefaultTask() {

    /** Package name -> pinned version, sourced from the `npm-*` catalog aliases. */
    @get:Input
    abstract val pins: MapProperty<String, String>

    @get:Input
    abstract val failOnMissingPackage: Property<Boolean>

    /** The committed lockfiles under `kotlin-js-store/`. Missing files are skipped. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val lockFiles: ConfigurableFileCollection

    /** Only used to render lockfile paths relative to the repository root in messages. */
    @get:Internal
    abstract val rootDirectory: DirectoryProperty

    @TaskAction
    fun verify() {
        val expected = pins.get()
        if (expected.isEmpty()) {
            logger.lifecycle("[npm-pins] No npm pins declared; nothing to verify.")
            return
        }

        val mismatches = mutableListOf<String>()
        val seen = mutableSetOf<String>()
        val rootDir = rootDirectory.get().asFile

        lockFiles.files.filter { it.isFile }.sortedBy { it.path }.forEach { lockFile ->
            val relative = lockFile.relativeToOrSelf(rootDir).path
            YarnLockParser.parse(lockFile.readText()).forEach { (packageName, version) ->
                val pinned = expected[packageName] ?: return@forEach
                seen += packageName
                if (version != pinned) {
                    mismatches += "$relative: $packageName resolved to $version, pinned to $pinned"
                }
            }
        }

        val missing = expected.keys - seen
        if (missing.isNotEmpty()) {
            val message = "[npm-pins] Pinned but absent from every lockfile: ${missing.sorted().joinToString(", ")}. " +
                    "The pin may be stale — drop it from gradle/libs.versions.toml if the package is gone for good."
            if (failOnMissingPackage.get()) throw GradleException(message)
            logger.warn(message)
        }

        if (mismatches.isNotEmpty()) {
            throw GradleException(
                buildString {
                    appendLine("[npm-pins] Lockfile does not honour the declared npm pins:")
                    mismatches.sorted().forEach { appendLine("  - $it") }
                    appendLine()
                    appendLine("Do not edit kotlin-js-store/**/yarn.lock by hand — it is regenerated. Instead run:")
                    appendLine("  ./gradlew kotlinUpgradeYarnLock kotlinWasmUpgradeYarnLock")
                    append("and commit the regenerated lockfiles.")
                }
            )
        }

        logger.lifecycle("[npm-pins] ${expected.size} pin(s) verified against ${lockFiles.files.count { it.isFile }} lockfile(s).")
    }
}
