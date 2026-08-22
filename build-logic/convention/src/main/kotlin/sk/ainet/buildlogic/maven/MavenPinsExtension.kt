package sk.ainet.buildlogic.maven

import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider

/**
 * The `mavenPins { }` block nested inside the root `skainet { }` extension.
 *
 * Every pin names a Maven/JVM coordinate literally and takes its version from the
 * version catalog, so the number stays in one place and stays bumpable by tooling:
 *
 * ```kotlin
 * // gradle/libs.versions.toml
 * [versions]
 * maven-netty = "4.1.136.Final"   # CVE-2026-56819
 *
 * // root build.gradle.kts
 * skainet {
 *     mavenPins {
 *         pin("io.netty:netty-handler", libs.versions.maven.netty)
 *     }
 * }
 * ```
 *
 * Unlike npm pins, a Maven pin applies to the single dependency graph shared by every
 * subproject — there is no per-target lockfile to scope it to.
 */
abstract class MavenPinsExtension {

    /**
     * `"group:artifact"` -> exact version. Consumed by [MavenPinsPlugin] and
     * [VerifyMavenPinsTask]; declare pins through [pin] rather than mutating this
     * directly, which skips validation and duplicate detection.
     */
    abstract val pins: MapProperty<String, String>

    /**
     * Whether `verifyMavenPins` fails when a pinned coordinate never appears in any
     * subproject's resolved dependency graph. Defaults to `false`: a pin that outlives
     * the dependency that once pulled it in transitively is stale rather than broken,
     * and should be reported without breaking the build.
     */
    abstract val failOnMissingModule: Property<Boolean>

    private val declared = mutableSetOf<String>()

    /**
     * Pins [coordinate] (`"group:artifact"`, e.g. `"io.netty:netty-handler"`) to a
     * version held in the version catalog.
     */
    fun pin(coordinate: String, version: Provider<String>) {
        val name = validateCoordinate(coordinate)
        val checked = version.map { validateVersion(name, it) }
        pins.put(name, checked)
    }

    /**
     * Pins [coordinate] to a literal version.
     *
     * Prefer the [Provider] overload — a number in `libs.versions.toml` is visible to
     * dependency-update tooling, a number in the build script is not.
     */
    fun pin(coordinate: String, version: String) {
        val name = validateCoordinate(coordinate)
        val checked = validateVersion(name, version)
        pins.put(name, checked)
    }

    private fun validateCoordinate(coordinate: String): String {
        val name = coordinate.trim()
        require(name.isNotEmpty()) { "[maven-pins] Coordinate must not be blank" }
        require(name.none { it.isWhitespace() }) {
            "[maven-pins] Coordinate '$coordinate' must not contain whitespace"
        }
        require(name.count { it == ':' } == 1 && !name.startsWith(":") && !name.endsWith(":")) {
            "[maven-pins] Coordinate '$coordinate' must be exactly \"group:artifact\" " +
                    "(no version — that goes in the version argument)"
        }
        require(declared.add(name)) {
            "[maven-pins] '$name' is pinned twice — declare it once."
        }
        return name
    }

    /**
     * Rejects ranges. `verifyMavenPins` compares the resolved version for exact
     * equality, so a range pin such as `4.1.+` could never verify — better to fail at
     * configuration time with the reason than at `check` with a confusing mismatch.
     */
    private fun validateVersion(coordinate: String, version: String): String {
        val exact = version.trim()
        require(exact.isNotEmpty()) {
            "[maven-pins] Pin for '$coordinate' must declare a version (e.g. \"4.1.136.Final\")"
        }
        require(exact.none { it in RANGE_CHARACTERS }) {
            "[maven-pins] Pin for '$coordinate' must be an exact version, but was '$exact'. " +
                    "Ranges cannot be verified — write \"4.1.136.Final\", not \"4.1.+\"."
        }
        return exact
    }

    private companion object {
        private val RANGE_CHARACTERS = setOf('^', '~', '>', '<', '=', '*', '+', '|', ' ')
    }
}
