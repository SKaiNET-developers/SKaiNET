package sk.ainet.buildlogic.npm

import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property

/**
 * Configuration for `sk.ainet.npm-pins`.
 *
 * The pins themselves are **not** declared here — they live in `[versions]` of
 * `gradle/libs.versions.toml` under the `npm-` prefix, so a package is pinned by
 * editing exactly one line:
 *
 * ```toml
 * [versions]
 * npm-ws = "8.21.1"   # GHSA-96hv-2xvq-fx4p
 * ```
 *
 * This extension only carries the knobs needed to interpret that catalog.
 */
abstract class NpmPinsExtension {

    /** Version catalog to read pins from. Defaults to `libs`. */
    abstract val catalogName: Property<String>

    /** Alias prefix marking a `[versions]` entry as an npm pin. Defaults to `npm`. */
    abstract val aliasPrefix: Property<String>

    /**
     * Escape hatch for package names the alias rule cannot express.
     *
     * Gradle normalises catalog aliases, so `npm-webpack-dev-server` arrives as
     * `npm.webpack.dev.server` and maps back to `webpack-dev-server` by turning the
     * separators into dashes. Packages whose real name contains a dot (`socket.io`)
     * or a scope (`@types/node`) round-trip incorrectly and need an explicit entry,
     * keyed by the normalised alias:
     *
     * ```kotlin
     * npmPins {
     *     packageNameOverrides.put("npm.socket.io", "socket.io")
     * }
     * ```
     */
    abstract val packageNameOverrides: MapProperty<String, String>

    /**
     * Whether `verifyNpmPins` fails when a pinned package is absent from every
     * lockfile. Defaults to `false`: a pin that outlives its package (because the
     * Kotlin toolchain dropped the dependency) is stale rather than broken, and
     * should be reported without breaking the build.
     */
    abstract val failOnMissingPackage: Property<Boolean>
}
