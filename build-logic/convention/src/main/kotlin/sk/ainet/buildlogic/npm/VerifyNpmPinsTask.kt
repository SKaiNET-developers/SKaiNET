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
import java.io.File

/**
 * Fails when a committed Yarn lockfile disagrees with a declared npm pin.
 *
 * Yarn `resolutions` make the pin take effect, but only the next time the lockfile
 * is regenerated. Without this check a stale or hand-edited lockfile silently wins
 * — which is exactly how PR #894's `ws` bump ended up as a zero-line diff.
 *
 * Each lockfile is checked against its own pin map, so a pin scoped to one
 * [NpmPinTarget] is never reported as missing from the other lockfile.
 */
@DisableCachingByDefault(because = "Reads two small lockfiles; caching costs more than it saves")
abstract class VerifyNpmPinsTask : DefaultTask() {

    /** Package name -> pinned version for the Kotlin/JS lockfile. */
    @get:Input
    abstract val jsPins: MapProperty<String, String>

    /** Package name -> pinned version for the Kotlin/Wasm lockfile. */
    @get:Input
    abstract val wasmPins: MapProperty<String, String>

    @get:Input
    abstract val failOnMissingPackage: Property<Boolean>

    /** `kotlin-js-store/yarn.lock`. Missing file is skipped. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val jsLockFile: ConfigurableFileCollection

    /** `kotlin-js-store/wasm/yarn.lock`. Missing file is skipped. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val wasmLockFile: ConfigurableFileCollection

    /** Only used to render lockfile paths relative to the repository root in messages. */
    @get:Internal
    abstract val rootDirectory: DirectoryProperty

    @TaskAction
    fun verify() {
        val lanes = listOf(
            Lane("Kotlin/JS", jsPins.get(), jsLockFile.files),
            Lane("Kotlin/Wasm", wasmPins.get(), wasmLockFile.files),
        )

        val pinnedPackages = lanes.flatMap { it.pins.keys }.toSortedSet()
        if (pinnedPackages.isEmpty()) {
            logger.lifecycle("[npm-pins] No npm pins declared; nothing to verify.")
            return
        }

        val rootDir = rootDirectory.get().asFile
        val mismatches = mutableListOf<String>()
        val missing = mutableListOf<String>()
        var checkedLockFiles = 0

        for (lane in lanes) {
            if (lane.pins.isEmpty()) continue
            val lockFile = lane.lockFiles.firstOrNull { it.isFile } ?: continue
            checkedLockFiles++

            val relative = lockFile.relativeToOrSelf(rootDir).path
            val seen = mutableSetOf<String>()
            YarnLockParser.parse(lockFile.readText()).forEach { (packageName, version) ->
                val pinned = lane.pins[packageName] ?: return@forEach
                seen += packageName
                if (version != pinned) {
                    mismatches += "$relative: $packageName resolved to $version, pinned to $pinned"
                }
            }
            (lane.pins.keys - seen).sorted().forEach { missing += "$relative (${lane.label}): $it" }
        }

        if (missing.isNotEmpty()) {
            val message = buildString {
                appendLine("[npm-pins] Pinned but absent from the lockfile the pin is scoped to:")
                missing.sorted().forEach { appendLine("  - $it") }
                append(
                    "The pin may be stale — drop it from gradle/libs.versions.toml if the " +
                            "package is gone for good, or narrow its NpmPinTarget if it only ever " +
                            "existed in the other graph."
                )
            }
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

        logger.lifecycle(
            "[npm-pins] ${pinnedPackages.size} pin(s) verified against $checkedLockFiles lockfile(s)."
        )
    }

    private data class Lane(val label: String, val pins: Map<String, String>, val lockFiles: Set<File>)
}
