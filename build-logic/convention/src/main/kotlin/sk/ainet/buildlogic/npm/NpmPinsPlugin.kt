package sk.ainet.buildlogic.npm

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.artifacts.VersionCatalog
import org.gradle.api.artifacts.VersionCatalogsExtension
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnLockStoreTask
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension
import org.jetbrains.kotlin.gradle.targets.wasm.yarn.WasmYarnPlugin
import org.jetbrains.kotlin.gradle.targets.wasm.yarn.WasmYarnRootExtension
import org.jetbrains.kotlin.gradle.targets.web.yarn.BaseYarnRootExtension

/**
 * Pins selected npm packages to an audited version across every Kotlin/JS and
 * Kotlin/Wasm lockfile in the build.
 *
 * ## Why this exists
 *
 * `kotlin-js-store/yarn.lock` and `kotlin-js-store/wasm/yarn.lock` are generated —
 * editing them by hand does not survive the next `kotlinUpgradeYarnLock`. Nor does
 * declaring `implementation(npm("ws", "8.21.1"))` in a source set help when the
 * package is pulled in transitively: the transitive ranges still resolve on their
 * own. The mechanism that actually constrains the graph is Yarn `resolutions`, which
 * the Kotlin Gradle plugin writes into the generated root `package.json` from the
 * Yarn root extension.
 *
 * There are two such extensions — one for JS, one for Wasm — each with its own
 * lockfile, so a pin has to be applied twice. This plugin does that from a single
 * declaration.
 *
 * ## Declaring a pin
 *
 * Add one line to `[versions]` in `gradle/libs.versions.toml`, prefixed with `npm-`:
 *
 * ```toml
 * npm-ws = "8.21.1"   # GHSA-96hv-2xvq-fx4p
 * ```
 *
 * Then regenerate and commit the lockfiles:
 *
 * ```
 * ./gradlew kotlinUpgradeYarnLock kotlinWasmUpgradeYarnLock
 * ```
 *
 * `verifyNpmPins` (wired into `check`) fails if a lockfile later drifts off a pin.
 *
 * Must be applied to the root project: the Yarn root extensions assert that they
 * belong to `rootProject`.
 */
class NpmPinsPlugin : Plugin<Project> {

    override fun apply(project: Project) {
        require(project == project.rootProject) {
            "[npm-pins] sk.ainet.npm-pins must be applied to the root project — " +
                    "Yarn roots are root-project scoped, but it was applied to ${project.path}"
        }

        val extension = project.extensions.create("npmPins", NpmPinsExtension::class.java).apply {
            catalogName.convention("libs")
            aliasPrefix.convention("npm")
            packageNameOverrides.convention(emptyMap())
            failOnMissingPackage.convention(false)
        }

        // Resolved lazily so overrides declared later in the root build script are seen.
        val pinsProvider: Provider<Map<String, String>> = project.provider { readPins(project, extension) }

        // The Yarn plugins are applied by KGP only once a js/wasmJs target is configured,
        // which happens well after this plugin is applied — hence the reactive hooks.
        project.plugins.withType(YarnPlugin::class.java) {
            project.extensions.getByType(YarnRootExtension::class.java).applyPins(pinsProvider.get())
        }
        project.plugins.withType(WasmYarnPlugin::class.java) {
            // getByName rather than WasmYarnRootExtension[project]: the latter applies
            // WasmYarnPlugin as a side effect, dragging wasm Yarn setup into js-only builds.
            val wasmYarn = project.extensions.getByName(WasmYarnRootExtension.YARN) as WasmYarnRootExtension
            wasmYarn.applyPins(pinsProvider.get())
        }

        val verify = project.tasks.register("verifyNpmPins", VerifyNpmPinsTask::class.java) {
            group = "verification"
            description = "Check the committed Yarn lockfiles against the npm-* pins in the version catalog"
            pins.set(pinsProvider)
            failOnMissingPackage.set(extension.failOnMissingPackage)
            rootDirectory.set(project.layout.projectDirectory)
            lockFiles.from(
                project.layout.projectDirectory.file("kotlin-js-store/yarn.lock"),
                project.layout.projectDirectory.file("kotlin-js-store/wasm/yarn.lock"),
            )
            // The lockfiles are outputs of the store tasks. Order after them rather than
            // depending on them, so `verifyNpmPins` stays a cheap file check on its own but
            // still sees the current state when a full build refreshes the lockfiles.
            mustRunAfter(project.tasks.withType(YarnLockStoreTask::class.java))
        }

        project.pluginManager.withPlugin("base") {
            project.tasks.named("check").configure { dependsOn(verify) }
        }
    }

    private fun BaseYarnRootExtension.applyPins(pins: Map<String, String>) {
        pins.forEach { (packageName, version) -> resolution(packageName, version) }
    }

    private fun readPins(project: Project, extension: NpmPinsExtension): Map<String, String> {
        val catalogName = extension.catalogName.get()
        val catalog: VersionCatalog = project.extensions
            .getByType(VersionCatalogsExtension::class.java)
            .find(catalogName)
            .orElseThrow {
                IllegalStateException("[npm-pins] Version catalog '$catalogName' not found")
            }

        val prefix = extension.aliasPrefix.get()
        val overrides = extension.packageNameOverrides.get()

        return catalog.versionAliases
            .filter { it.normalizeSeparators().startsWith("$prefix.") }
            .associate { alias ->
                val packageName = overrides[alias.normalizeSeparators()] ?: alias.toPackageName(prefix)
                val version = catalog.findVersion(alias)
                    .orElseThrow { IllegalStateException("[npm-pins] Version alias '$alias' disappeared from '$catalogName'") }
                    .requiredVersion
                require(version.isNotEmpty()) {
                    "[npm-pins] Catalog alias '$alias' must declare an exact version (e.g. npm-ws = \"8.21.1\")"
                }
                packageName to version
            }
    }

    /**
     * `npm-webpack-dev-server` reaches us as `npm.webpack.dev.server`; strip the prefix
     * and turn the remaining separators back into dashes.
     *
     * Package names that genuinely contain a dot (`socket.io`) or a scope
     * (`@types/node`) cannot be expressed as a catalog alias and need an entry in
     * [NpmPinsExtension.packageNameOverrides].
     */
    private fun String.toPackageName(prefix: String): String =
        normalizeSeparators().removePrefix("$prefix.").replace('.', '-')

    private fun String.normalizeSeparators(): String = replace('-', '.').replace('_', '.')
}
