package sk.ainet.buildlogic.maven

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.provider.Provider
import sk.ainet.buildlogic.root.SkainetRootExtension

/**
 * Pins selected Maven/JVM coordinates to an audited version across every subproject's
 * dependency graph.
 *
 * ## Why this exists
 *
 * A high-severity CVE frequently lands in a package nobody declares directly — it is
 * pulled in transitively by something else, and the direct dependency that causes it
 * may itself have no newer release. Gradle's `resolutionStrategy` is the mechanism
 * that actually constrains the graph in that case; this plugin gives it the same
 * single-declaration, catalog-sourced shape that `sk.ainet.npm-pins` gives Yarn
 * `resolutions`, instead of a bespoke `resolutionStrategy` block being reinvented in
 * the root build script each time.
 *
 * ## Declaring a pin
 *
 * Put the version in `[versions]` of `gradle/libs.versions.toml` and name the
 * coordinate in the root build script:
 *
 * ```toml
 * maven-netty = "4.1.136.Final"   # CVE-2026-56819
 * ```
 *
 * ```kotlin
 * skainet {
 *     mavenPins {
 *         pin("io.netty:netty-handler", libs.versions.maven.netty)
 *     }
 * }
 * ```
 *
 * `verifyMavenPins` (wired into `check`) fails if a pinned coordinate stops resolving
 * to its declared version anywhere in the build. It is registered per subproject —
 * Gradle only allows a task to resolve configurations that belong to its own project,
 * so a single root-level verify task cannot walk every subproject's classpath itself.
 * Running `./gradlew verifyMavenPins` from the repository root still verifies the
 * whole build: Gradle's CLI matches a bare task name against every project.
 *
 * Must be applied to the root project: pins are declared once, in the root `skainet {}`
 * block, and — like `sk.ainet.npm-pins` — must be declared while the root script is
 * evaluated, before any subproject reads them.
 */
class MavenPinsPlugin : Plugin<Project> {

    override fun apply(project: Project) {
        require(project == project.rootProject) {
            "[maven-pins] sk.ainet.maven-pins must be applied to the root project — " +
                    "pins are declared once in the root skainet { } block, but it was applied to ${project.path}"
        }

        val extension = SkainetRootExtension.findOrCreate(project).mavenPins.apply {
            failOnMissingModule.convention(false)
        }

        // Resolved lazily inside eachDependency, which only fires when a configuration
        // is actually resolved — by then the root script body (where pins are declared)
        // has long finished evaluating.
        val pins: Provider<Map<String, String>> = extension.pins

        project.allprojects {
            configurations.all {
                resolutionStrategy.eachDependency {
                    val coordinate = "${requested.group}:${requested.name}"
                    pins.get()[coordinate]?.let { pinned -> useVersion(pinned) }
                }
            }
        }

        // One verify task per subproject, each resolving only its own configurations.
        // The root project never has *CompileClasspath/*RuntimeClasspath configurations
        // of its own, so it gets no task — an empty one would just print a spurious
        // "pin never resolved" warning for pins that every subproject resolves fine.
        project.subprojects {
            val verify = tasks.register("verifyMavenPins", VerifyMavenPinsTask::class.java) {
                group = "verification"
                description = "Check that every coordinate declared in the root skainet { mavenPins { } } " +
                        "actually resolves to its pinned version in this subproject"
                this.pins.set(pins)
                failOnMissingModule.set(extension.failOnMissingModule)
            }

            // `configureEach` rather than a one-shot lookup: most of a subproject's
            // configurations (especially KMP per-target ones) don't exist yet when this
            // plugin applies to the root project, only once the subproject's own build
            // script evaluates.
            //
            // Scoped to *CompileClasspath/*RuntimeClasspath: every pinned coordinate here
            // is a JVM-only library, so these are the only configurations that could ever
            // resolve one. This also sidesteps the various non-classpath resolvable
            // configurations (coverage/report aggregation buckets and the like) some
            // plugins wire up in ways `isCanBeResolved` alone doesn't reliably describe
            // at configuration time.
            configurations.matching { configuration ->
                configuration.isCanBeResolved &&
                        (configuration.name.endsWith("CompileClasspath") || configuration.name.endsWith("RuntimeClasspath"))
            }.configureEach {
                val configuration = this
                verify.configure { configurations.add(configuration) }
            }

            pluginManager.withPlugin("base") {
                tasks.named("check").configure { dependsOn(verify) }
            }
        }
    }
}
