package sk.ainet.buildlogic.npm

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.provider.Provider
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnLockStoreTask
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnPlugin
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension
import org.jetbrains.kotlin.gradle.targets.wasm.yarn.WasmYarnPlugin
import org.jetbrains.kotlin.gradle.targets.wasm.yarn.WasmYarnRootExtension
import org.jetbrains.kotlin.gradle.targets.web.yarn.BaseYarnRootExtension
import sk.ainet.buildlogic.root.SkainetRootExtension

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
 * declaration, and lets a pin opt out of one of them via [NpmPinTarget] when the
 * package exists in only one graph.
 *
 * ## Declaring a pin
 *
 * Put the version in `[versions]` of `gradle/libs.versions.toml` and name the package
 * in the root build script:
 *
 * ```toml
 * npm-ws = "8.21.1"   # GHSA-96hv-2xvq-fx4p
 * ```
 *
 * ```kotlin
 * skainet {
 *     npmPins {
 *         pin("ws", libs.versions.npm.ws)
 *         // Only the JS graph bundles with webpack; scoping keeps webpack's ~76
 *         // transitive packages out of kotlin-js-store/wasm/yarn.lock.
 *         pin("webpack", libs.versions.npm.webpack, NpmPinTarget.JS)
 *     }
 * }
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
 * belong to `rootProject`, and pins must be declared while the root script is
 * evaluated, before any module configures a web target.
 */
class NpmPinsPlugin : Plugin<Project> {

    override fun apply(project: Project) {
        require(project == project.rootProject) {
            "[npm-pins] sk.ainet.npm-pins must be applied to the root project — " +
                    "Yarn roots are root-project scoped, but it was applied to ${project.path}"
        }

        val extension = SkainetRootExtension.findOrCreate(project).npmPins.apply {
            failOnMissingPackage.convention(false)
        }

        // Lets web-targeted modules assert that pins are in force; see NpmPinsMarker.
        NpmPinsMarker.register(project)

        // Resolved lazily: pins are declared in the root script body, which runs after
        // the plugins block that applies this plugin.
        val jsPins: Provider<Map<String, String>> = extension.jsPins
        val wasmPins: Provider<Map<String, String>> = extension.wasmPins

        // The Yarn plugins are applied by KGP only once a js/wasmJs target is configured,
        // which happens well after this plugin is applied — hence the reactive hooks.
        // Each root gets only the pins scoped to it; see NpmPinTarget for why that matters.
        project.plugins.withType(YarnPlugin::class.java) {
            project.extensions.getByType(YarnRootExtension::class.java).applyPins(jsPins.get())
        }
        project.plugins.withType(WasmYarnPlugin::class.java) {
            // getByName rather than WasmYarnRootExtension[project]: the latter applies
            // WasmYarnPlugin as a side effect, dragging wasm Yarn setup into js-only builds.
            val wasmYarn = project.extensions.getByName(WasmYarnRootExtension.YARN) as WasmYarnRootExtension
            wasmYarn.applyPins(wasmPins.get())
        }

        val verify = project.tasks.register("verifyNpmPins", VerifyNpmPinsTask::class.java) {
            group = "verification"
            description = "Check the committed Yarn lockfiles against the pins declared in skainet { npmPins { } }"
            this.jsPins.set(jsPins)
            this.wasmPins.set(wasmPins)
            failOnMissingPackage.set(extension.failOnMissingPackage)
            rootDirectory.set(project.layout.projectDirectory)
            jsLockFile.from(project.layout.projectDirectory.file("kotlin-js-store/yarn.lock"))
            wasmLockFile.from(project.layout.projectDirectory.file("kotlin-js-store/wasm/yarn.lock"))
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
}
