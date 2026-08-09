package sk.ainet.buildlogic.npm

import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.provider.Provider

/**
 * The `npmPins { }` block nested inside the root `skainet { }` extension.
 *
 * Every pin names its package literally and takes its version from the version
 * catalog, so the number stays in one place and stays bumpable by tooling:
 *
 * ```kotlin
 * // gradle/libs.versions.toml
 * [versions]
 * npm-ws = "8.21.1"   # GHSA-96hv-2xvq-fx4p
 *
 * // root build.gradle.kts
 * skainet {
 *     npmPins {
 *         pin("ws", libs.versions.npm.ws)
 *     }
 * }
 * ```
 *
 * Because the package name is written out rather than derived from the alias, names
 * that a catalog alias cannot spell — a dot (`socket.io`), a scope (`@types/node`) —
 * need nothing special:
 *
 * ```kotlin
 * pin("socket.io", libs.versions.npm.socketio)
 * pin("@types/node", libs.versions.npm.typesNode)
 * ```
 *
 * ## Scoping a pin to one lockfile
 *
 * A pin applies to both Yarn roots unless told otherwise. Pass [NpmPinTarget] values
 * when the package lives in only one graph, so the other lockfile does not acquire an
 * entry — and a whole transitive tree — for a package it never resolves:
 *
 * ```kotlin
 * pin("webpack", libs.versions.npm.webpack, NpmPinTarget.JS)
 * ```
 */
abstract class NpmPinsExtension {

    /**
     * Package name -> exact version for `kotlin-js-store/yarn.lock`. Consumed by
     * [NpmPinsPlugin] and [VerifyNpmPinsTask]; declare pins through [pin] rather than
     * mutating this directly, which skips validation and duplicate detection.
     */
    abstract val jsPins: MapProperty<String, String>

    /** Package name -> exact version for `kotlin-js-store/wasm/yarn.lock`. See [jsPins]. */
    abstract val wasmPins: MapProperty<String, String>

    /**
     * Whether `verifyNpmPins` fails when a pinned package is absent from the lockfile
     * it is scoped to. Defaults to `false`: a pin that outlives its package (because
     * the Kotlin toolchain dropped the dependency) is stale rather than broken, and
     * should be reported without breaking the build.
     */
    abstract val failOnMissingPackage: Property<Boolean>

    private val declared = mutableSetOf<String>()

    /**
     * Pins [packageName] to a version held in the version catalog.
     *
     * With no [targets] the pin applies to every lockfile; name them to restrict it.
     */
    fun pin(packageName: String, version: Provider<String>, vararg targets: NpmPinTarget) {
        val name = validatePackageName(packageName)
        val checked = version.map { validateVersion(name, it) }
        scopeOf(targets).forEach { target -> mapFor(target).put(name, checked) }
    }

    /**
     * Pins [packageName] to a literal version.
     *
     * Prefer the [Provider] overload — a number in `libs.versions.toml` is visible to
     * dependency-update tooling, a number in the build script is not.
     */
    fun pin(packageName: String, version: String, vararg targets: NpmPinTarget) {
        val name = validatePackageName(packageName)
        val checked = validateVersion(name, version)
        scopeOf(targets).forEach { target -> mapFor(target).put(name, checked) }
    }

    private fun scopeOf(targets: Array<out NpmPinTarget>): Set<NpmPinTarget> =
        if (targets.isEmpty()) NpmPinTarget.ALL else targets.toSet()

    private fun mapFor(target: NpmPinTarget): MapProperty<String, String> = when (target) {
        NpmPinTarget.JS -> jsPins
        NpmPinTarget.WASM -> wasmPins
    }

    private fun validatePackageName(packageName: String): String {
        val name = packageName.trim()
        require(name.isNotEmpty()) { "[npm-pins] Package name must not be blank" }
        require(name.none { it.isWhitespace() }) {
            "[npm-pins] Package name '$packageName' must not contain whitespace"
        }
        require(declared.add(name)) {
            "[npm-pins] '$name' is pinned twice — a Yarn resolution is global within a " +
                    "Yarn root, so the second declaration would silently win. Declare it " +
                    "once and pass both targets, or scope each pin to a different target."
        }
        return name
    }

    /**
     * Rejects ranges. `verifyNpmPins` compares the lockfile's resolved version for exact
     * equality, so a range pin such as `^8.21.1` could never verify — better to fail at
     * configuration time with the reason than at `check` with a confusing mismatch.
     */
    private fun validateVersion(packageName: String, version: String): String {
        val exact = version.trim()
        require(exact.isNotEmpty()) {
            "[npm-pins] Pin for '$packageName' must declare a version (e.g. \"8.21.1\")"
        }
        require(exact.none { it in RANGE_CHARACTERS }) {
            "[npm-pins] Pin for '$packageName' must be an exact version, but was '$exact'. " +
                    "Ranges cannot be verified against a lockfile — write \"8.21.1\", not \"^8.21.1\"."
        }
        return exact
    }

    private companion object {
        private val RANGE_CHARACTERS = setOf('^', '~', '>', '<', '=', '*', '|', ' ')
    }
}
